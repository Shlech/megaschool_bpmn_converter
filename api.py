import io
import os
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from sahi.predict import get_sliced_prediction

from core import (
    load_detection_models,
    load_ocr_model,
    clean_lane_overlaps,
    run_ocr_on_detections,
    get_lane_names_final,
    generate_bpmn_markdown,
    generate_simple_markdown,
    BPMN_CLASSES,
    CLASS_MAPPING,
    LANE_CLASS_ID,
)

app = FastAPI(title="Diagram2MD API (Queue)")

# -----------------------------
# Настройки очереди/воркеров
# -----------------------------
QUEUE_MAXSIZE = int(os.getenv("QUEUE_MAXSIZE", "100"))
WORKERS = int(os.getenv("WORKERS", "8"))
RESULT_TTL_SEC = int(os.getenv("RESULT_TTL_SEC", str(60 * 30)))  # 30 минут

MAX_MB = int(os.getenv("MAX_MB", "10"))

# -----------------------------
# Глобальные модели (грузим один раз)
# -----------------------------
# bpmn_object_model -> BPMN блоки
# flow_object_model -> обычные (flow) диаграммы
# arrow_model       -> стрелки
bpmn_object_model, flow_object_model, arrow_detection_model = load_detection_models()
ocr_object_model, ocr_lane_model = load_ocr_model()

# -----------------------------
# Очередь и хранилище статусов
# -----------------------------
job_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

@dataclass
class JobState:
    job_id: str
    filename: str
    status: str = "queued"  # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None  # {"markdown": "...", ...}

jobs: Dict[str, JobState] = {}
jobs_lock = asyncio.Lock()


# -----------------------------
# Внутренняя обработка 1 задачи
# -----------------------------
def _process_one_image_bytes(content: bytes, filename: str, labels_csv: str, diagram_type: str) -> Dict[str, Any]:
    # 0) Декод картинки
    img = Image.open(io.BytesIO(content)).convert("RGB")
    image_np = np.array(img)

    # -----------------------------
    # Обычные диаграммы (flow)
    # -----------------------------
    if diagram_type == "simple":
        # 1) DETECT OBJECTS (flow_model.pt)
        obj_res = get_sliced_prediction(
            image_np, flow_object_model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.3, overlap_width_ratio=0.3,
            perform_standard_pred=True,
            postprocess_type='NMS',
            postprocess_match_threshold=0.3
        )
        objects_coco = obj_res.to_coco_predictions()

        # 2) OCR объектов (берём все классы, кроме lane/pool если они есть)
        # SAHI объект: det.category.id, det.category.name
        target_class_ids = sorted({
            det.category.id for det in obj_res.object_prediction_list
            if getattr(det, "category", None) is not None and det.category.id != LANE_CLASS_ID
        })

        ocr_objects = run_ocr_on_detections(
            image_np,
            obj_res.object_prediction_list,
            ocr_object_model,
            target_class_ids
        )

        # 3) DETECT ARROWS (общая модель стрелок)
        arrow_res = get_sliced_prediction(
            image_np, arrow_detection_model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.4, overlap_width_ratio=0.4,
            postprocess_type='NMM',
            postprocess_match_threshold=0.2,
            verbose=0
        )
        arrows_coco = arrow_res.to_coco_predictions()

        # 4) MARKDOWN
        md = generate_simple_markdown(objects_coco, arrows_coco, ocr_objects)

        return {
            "filename": filename,
            "diagram_type": "simple",
            "markdown": md,
        }

    # 1) Классы
    if labels_csv:
        selected_labels = [x.strip() for x in labels_csv.split(",") if x.strip() in BPMN_CLASSES]
    else:
        selected_labels = ['task', 'event', 'exclusiveGateway', 'parallelGateway', 'lane']
        selected_labels = [c for c in selected_labels if c in BPMN_CLASSES]

    target_class_ids = [CLASS_MAPPING[label] for label in selected_labels]

    # 2) DETECT OBJECTS
    object_result = get_sliced_prediction(
        image_np, bpmn_object_model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.3, overlap_width_ratio=0.3,
        perform_standard_pred=True,
        postprocess_type='NMS',
        postprocess_match_threshold=0.3
    )

    # фильтруем и чистим lane
    object_result.object_prediction_list = [
        obj for obj in object_result.object_prediction_list
        if obj.category.id in target_class_ids or obj.category.id == LANE_CLASS_ID
    ]
    object_result.object_prediction_list = clean_lane_overlaps(
        object_result.object_prediction_list, LANE_CLASS_ID
    )
    object_predictions_coco = object_result.to_coco_predictions()

    # 3) OCR объектов
    ocr_results_data = run_ocr_on_detections(
        image_np,
        object_result.object_prediction_list,
        ocr_object_model,
        target_class_ids
    )

    # 4) OCR лейнов
    lane_results = get_lane_names_final(image_np, ocr_lane_model)

    # 5) DETECT ARROWS
    arrow_result = get_sliced_prediction(
        image_np, arrow_detection_model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.4, overlap_width_ratio=0.4,
        postprocess_type='NMM',
        postprocess_match_threshold=0.2,
        verbose=0
    )
    arrow_predictions_coco = arrow_result.to_coco_predictions()

    # 6) MARKDOWN
    if diagram_type == "bpmn":
        md = generate_bpmn_markdown(
            object_predictions_coco,
            arrow_predictions_coco,
            ocr_results_data,
            lane_results
        )
    else:
        md = generate_simple_markdown(
            object_predictions_coco,
            arrow_predictions_coco,
            ocr_results_data
        )

    return {
        "filename": filename,
        "markdown": md,
        "selected_labels": selected_labels,
        "diagram_type": diagram_type,
    }


async def _run_job(job_id: str, content: bytes, filename: str, labels_csv: str, diagram_type: str):
    # обновляем статус
    async with jobs_lock:
        st = jobs.get(job_id)
        if not st:
            return
        st.status = "running"
        st.started_at = time.time()

    try:
        # ВАЖНО: это CPU-bound и блокирующее => выносим в thread
        result = await asyncio.to_thread(_process_one_image_bytes, content, filename, labels_csv, diagram_type)

        async with jobs_lock:
            st = jobs.get(job_id)
            if st:
                st.status = "done"
                st.result = result
                st.finished_at = time.time()

    except Exception as e:
        async with jobs_lock:
            st = jobs.get(job_id)
            if st:
                st.status = "error"
                st.error = str(e)
                st.finished_at = time.time()


async def worker_loop(worker_idx: int):
    while True:
        job = await job_queue.get()
        try:
            await _run_job(**job)
        finally:
            job_queue.task_done()


async def cleanup_loop():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        async with jobs_lock:
            to_del = []
            for job_id, st in jobs.items():
                if st.finished_at and (now - st.finished_at) > RESULT_TTL_SEC:
                    to_del.append(job_id)
            for job_id in to_del:
                del jobs[job_id]


@app.on_event("startup")
async def startup_event():
    # запускаем воркеры
    for i in range(WORKERS):
        asyncio.create_task(worker_loop(i))
    asyncio.create_task(cleanup_loop())


# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "workers": WORKERS, "queue_max": QUEUE_MAXSIZE}


@app.post("/submit")
async def submit_diagram(
    file: UploadFile = File(...),
    labels: str = Form(""),
    diagram_type: str = Form("bpmn"),  # <-- НОВОЕ
):

    # 1) тип файла
    if not (file.content_type or "").startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Upload an image file"})

    # 2) читаем bytes
    content = await file.read()

    # 3) лимит размера
    if len(content) > MAX_MB * 1024 * 1024:
        return JSONResponse(status_code=413, content={"error": f"File too large (>{MAX_MB}MB)"})

    # 4) очередь переполнена?
    if job_queue.full():
        return JSONResponse(status_code=429, content={"error": "Queue is full, try later"})

    job_id = uuid.uuid4().hex
    state = JobState(job_id=job_id, filename=file.filename)

    async with jobs_lock:
        jobs[job_id] = state

    # кладём задачу в очередь
    await job_queue.put({
        "job_id": job_id,
        "content": content,
        "filename": file.filename,
        "labels_csv": labels or "",
        "diagram_type": diagram_type,
    })

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    async with jobs_lock:
        st = jobs.get(job_id)
        if not st:
            return JSONResponse(status_code=404, content={"error": "job not found"})

        # позиция в очереди оценочно (только queued)
        queue_pos = None
        if st.status == "queued":
            # оценка: сколько queued раньше него (приблизительно)
            queue_pos = 1  # минимально; можно усложнять, но не обязательно для MVP

        return {
            "job_id": st.job_id,
            "filename": st.filename,
            "status": st.status,
            "created_at": st.created_at,
            "started_at": st.started_at,
            "finished_at": st.finished_at,
            "error": st.error,
            "queue_pos": queue_pos,
        }


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    async with jobs_lock:
        st = jobs.get(job_id)
        if not st:
            return JSONResponse(status_code=404, content={"error": "job not found"})
        if st.status == "queued" or st.status == "running":
            return JSONResponse(status_code=202, content={"status": st.status})
        if st.status == "error":
            return JSONResponse(status_code=500, content={"status": "error", "error": st.error})
        return {"status": "done", **(st.result or {})}
