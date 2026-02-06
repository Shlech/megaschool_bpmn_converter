import streamlit as st
import cv2
import numpy as np
import json
import os
import math
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from paddleocr import PaddleOCR

output_dir = 'st_tmp'
LANE_CLASS_ID = 6
# target_class_ids = [1, 2, 3, 6, 7, 13]
BPMN_CLASSES = [
    'dataAssociation', 'dataObject', 'dataStore', 'event',
    'eventBasedGateway', 'exclusiveGateway', 'lane', 'messageEvent',
    'messageFlow', 'parallelGateway', 'pool', 'sequenceFlow',
    'subProcess', 'task', 'timerEvent'
]
CLASS_MAPPING = {name: i for i, name in enumerate(BPMN_CLASSES)}
PADDING = {1: (0.4, 0.2), 2:(0.5, 0.2), 3: (1.5, 1.0), 7: (1.5, 1.0), 13: (0.01, 0.01)}

# Загрузка моделей
@st.cache_resource
def load_detection_models():
    """Загрузка моделей детекции. Замените на вашу реальную инициализацию"""
    object_model = AutoDetectionModel.from_pretrained(
        model_type='yolov11',
        model_path='models/object_weights.pt',
        confidence_threshold=0.4,
        device='cpu' 
    )
    arrow_model = AutoDetectionModel.from_pretrained(
        model_type='yolov11',
        model_path='models/arrow_weights.pt',
        confidence_threshold=0.5,
        device='cpu'
    )
    return object_model, arrow_model

@st.cache_resource
def load_ocr_model():
    """Загрузка PaddleOCR один раз на сессию"""
    object_ocr = PaddleOCR(
        ocr_version="PP-OCRv5",
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='cyrillic_PP-OCRv5_mobile_rec',
        lang='ru',
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=1.7,
        enable_hpi=True,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
        text_recognition_batch_size=32,
    )
    lane_ocr = PaddleOCR(
        ocr_version = "PP-OCRv5",
        text_detection_model_name = 'PP-OCRv5_mobile_det',
        text_recognition_model_name = 'cyrillic_PP-OCRv5_mobile_rec',
        lang = 'ru',
        text_det_thresh = 0.2,
        text_det_limit_side_len = 3000,
        text_det_box_thresh = 0.3,
        text_det_unclip_ratio = 2.2,
        enable_mkldnn = False,
        enable_hpi = True,
        textline_orientation_batch_size = 32,
        text_recognition_batch_size = 32
    )
    return object_ocr, lane_ocr

# Вспомогательные функции для детекции объектов
def calculate_iou(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def clean_lane_overlaps(predictions, lane_class_id, iou_threshold=0.2):
    lanes = [p for p in predictions if p.category.id == lane_class_id]
    others = [p for p in predictions if p.category.id != lane_class_id]
    
    if not lanes:
        return predictions

    # Сортируем по уверенности (от самых уверенных к менее)
    lanes.sort(key=lambda x: x.score.value, reverse=True)
    
    keep_lanes = []
    for candidate in lanes:
        is_overlap = False
        # Получаем координаты в формате [x1, y1, x2, y2]
        cand_box = candidate.bbox.to_xyxy() 
        
        for confirmed in keep_lanes:
            conf_box = confirmed.bbox.to_xyxy()
            
            if calculate_iou(cand_box, conf_box) > iou_threshold:
                is_overlap = True
                break
        
        if not is_overlap:
            keep_lanes.append(candidate)
            
    return keep_lanes + others
def preprocess_crop(crop, target_min_height=48):
    if crop is None or crop.size == 0:
        return crop
    processed = crop.copy()
    h, w = processed.shape[:2]
    if h < target_min_height:
        scale = target_min_height / h
        processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return processed
def run_ocr_on_detections(image_np, detections, ocr_model, target_class_ids):
    """
    Запускает OCR сразу на всех найденных объектах (Batch processing).
    Это быстрее и стабильнее, чем ThreadPoolExecutor для PaddleOCR v3.
    """
    crops = []
    valid_metadata = []
    
    # Конвертация в BGR для OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h_img, w_img = image_bgr.shape[:2]

    # 1. Сбор всех кропов
    for det in detections:
        # Адаптация под формат SAHI или COCO dict
        if hasattr(det, 'category'): # SAHI object
            cat_id = det.category.id
            bbox = det.bbox.to_xywh()
        else: # Dict
            cat_id = det['category_id']
            bbox = det['bbox']

        if cat_id in target_class_ids:
            x, y, w, h = map(int, bbox)
            px, py = PADDING.get(cat_id, (0, 0))
            pad_x, pad_y = int(w * px), int(h * py)
            
            y1, y2 = max(0, y - pad_y), min(h_img, y + h + pad_y)
            x1, x2 = max(0, x - pad_x), min(w_img, x + w + pad_x)
            
            crop = image_bgr[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Используем вашу функцию препроцессинга
                processed_crop = preprocess_crop(crop)
                crops.append(processed_crop)
                valid_metadata.append({
                    "class_id": cat_id,
                    "bbox": [x, y, w, h]
                })

    results = []
    
    # 2. Массовое предсказание (как в вашем скрипте)
    if crops:
        try:
            # Отправляем весь список кропов в модель
            outputs = ocr_model.predict(input=crops)
            
            # 3. Сопоставление результатов
            for i, res in enumerate(outputs):
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                
                detected_text = " ".join(texts)
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                results.append({
                    "class_id": valid_metadata[i]['class_id'],
                    "bbox": valid_metadata[i]['bbox'],
                    "text": detected_text.strip(),
                    "ocr_confidence": float(avg_score)
                })
        except Exception as e:
            st.error(f"Ошибка OCR: {e}")
            
    return results

def parse_bpmn_lanes_fixed(results, meta, y_threshold=100):
    """
    Парсит результаты OCR с повернутого изображения и возвращает координаты в исходной системе.
    Убрал сохранение файла внутри функции, чтобы делать это явно в main.
    """
    data = {}
    if isinstance(results, list):
        if len(results) > 0:
            data = results[0]
        else:
            return []
    elif isinstance(results, dict):
        data = results
    else:
        return []

    if 'rec_texts' not in data or 'dt_polys' not in data:
        return []

    texts = data['rec_texts']
    polys = data['dt_polys']
    scores = data.get('rec_scores', [])

    padding = meta['padding']
    scale = meta['scale_factor']
    h_rot = meta['rotated_h'] 

    items = []
    
    for i, text in enumerate(texts):
        if scores and scores[i] < 0.5: 
            continue
            
        poly = np.array(polys[i])
        
        # --- ОБРАТНАЯ ТРАНСФОРМАЦИЯ ---
        # 1. Убираем паддинг
        poly_unpad = poly - padding
        
        # 2. Обратный поворот (из 90 CW в оригинал)
        x_scaled = poly_unpad[:, 1]
        y_scaled = h_rot - poly_unpad[:, 0]
        
        # 3. Убираем масштаб
        x_orig = x_scaled / scale
        y_orig = y_scaled / scale
        
        items.append({
            'text': text, 
            'x': int(np.min(x_orig)), 
            'y': int(np.min(y_orig)),
            'bbox': [int(np.min(x_orig)), int(np.min(y_orig)), int(np.max(x_orig)), int(np.max(y_orig))]
        })

    if not items:
        return []

    # Сортировка и группировка
    items.sort(key=lambda k: k['y'])

    lanes_output = []
    if items:
        current_lane_words = [items[0]]
        for i in range(1, len(items)):
            prev = current_lane_words[-1]
            curr = items[i]
            if abs(curr['y'] - prev['y']) < (y_threshold / scale): 
                current_lane_words.append(curr)
            else:
                lanes_output.append(current_lane_words)
                current_lane_words = [curr]
        lanes_output.append(current_lane_words)

    final_data = []
    for group in lanes_output:
        full_text = " ".join([w['text'] for w in group])
        x1 = min(w['bbox'][0] for w in group)
        y1 = min(w['bbox'][1] for w in group)
        x2 = max(w['bbox'][2] for w in group)
        y2 = max(w['bbox'][3] for w in group)
        
        final_data.append({
            "lane_name": full_text,
            "bbox": [x1, y1, x2, y2],
            "type": "lane_label"
        })

    return final_data

def get_lane_names_final(image_np, ocr_model):
    """
    Принимает image_np (RGB) и модель OCR.
    """
    # Streamlit/PIL дают RGB, а OpenCV работает с BGR. Конвертируем.
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]

    # 1. Вырезаем левую часть
    search_width = int(w * 0.035) 
    left_margin = img[:, 0:search_width]
    
    if left_margin.size == 0:
        return []

    # Масштабирование
    scale_factor = 3
    left_margin_scaled = cv2.resize(
        left_margin, 
        None, 
        fx=scale_factor, 
        fy=scale_factor, 
        interpolation=cv2.INTER_CUBIC
    )
    
    # Поворот
    rotated_margin = cv2.rotate(left_margin_scaled, cv2.ROTATE_90_CLOCKWISE)
    
    h_rot, w_rot = rotated_margin.shape[:2]
    padding = int(min(h_rot, w_rot) * 0.106)
    padded_img = cv2.copyMakeBorder(rotated_margin, 
        padding, padding, padding, padding, 
        cv2.BORDER_CONSTANT, 
        value=(255, 255, 255)
    )
    
    meta = {
        "padding": padding,
        "scale_factor": scale_factor,
        "rotated_h": w_rot
    }
    
    try:
        results = ocr_model.predict(padded_img, use_textline_orientation=True)
        lane_data = parse_bpmn_lanes_fixed(results=results, meta=meta)
        return lane_data
    except Exception as e:
        print(f"Error in lane OCR: {e}")
        return []
import math
from collections import defaultdict

# -----------------------------
# 1. Утилиты (Ваш код)
# -----------------------------

def bbox_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)

def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def bbox_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    a_area = w1 * h1
    b_area = w2 * h2
    return inter_area / (a_area + b_area - inter_area + 1e-9)

# -----------------------------
# 2. Логика графа и геометрии
# -----------------------------
IGNORE_CLASSES = {"pool", "lane", "subProcess"}

def filter_nodes_for_graph(nodes):
    filtered = []
    for i, n in enumerate(nodes):
        # Проверка на наличие category_name (SAHI возвращает его)
        cls = n.get("category_name", str(n.get("category_id", "")))
        if cls in IGNORE_CLASSES:
            continue
        filtered.append((i, n))
    return filtered

def extract_by_class(nodes, cls_names):
    # cls_names может быть списком или set
    if isinstance(cls_names, str):
        cls_names = {cls_names}
    res = []
    for i, n in enumerate(nodes):
        cls = n.get("category_name", str(n.get("category_id", "")))
        if cls in cls_names:
            res.append((i, n))
    return res

def detect_orientation(node_items):
    centers = [bbox_center(n["bbox"]) for _, n in node_items]
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    spread_x = max(xs) - min(xs) if xs else 0
    spread_y = max(ys) - min(ys) if ys else 0
    return "horizontal" if spread_x >= spread_y else "vertical"

def arrow_endpoints(arrow_bbox):
    x, y, w, h = arrow_bbox
    if w >= h:
        start = (x, y + h / 2.0)
        end = (x + w, y + h / 2.0)
    else:
        start = (x + w / 2.0, y)
        end = (x + w / 2.0, y + h)
    return start, end

def bbox_edges(b):
    x, y, w, h = b
    cy = y + h / 2.0
    cx = x + w / 2.0
    return {
        "left": (x, cy), "right": (x + w, cy),
        "top": (cx, y), "bottom": (cx, y + h),
    }

def nearest_node(point, node_items):
    best = None
    best_d = float("inf")
    for node_id, n in node_items:
        edges = bbox_edges(n["bbox"])
        d = min(
            dist(point, edges["left"]), dist(point, edges["right"]),
            dist(point, edges["top"]), dist(point, edges["bottom"]),
        )
        if d < best_d:
            best_d = d
            best = node_id
    return best, best_d

def build_graph(node_items, arrows, max_link_dist=250):
    adj = defaultdict(list)
    indeg = defaultdict(int)
    for a in arrows:
        start, end = arrow_endpoints(a["bbox"])
        src, d1 = nearest_node(start, node_items)
        dst, d2 = nearest_node(end, node_items)
        if src is None or dst is None or src == dst:
            continue
        if d1 > max_link_dist or d2 > max_link_dist:
            continue
        adj[src].append(dst)
        indeg[dst] += 1
    return adj, indeg

def node_sort_key(node_id, id2node, orientation, eps=25):
    x, y = bbox_center(id2node[node_id]["bbox"])
    if orientation == "horizontal":
        return (round(x / eps), y, x)
    else:
        return (round(y / eps), x, y)

def restore_order(node_items, adj, indeg, orientation):
    id2node = {nid: n for nid, n in node_items}
    node_ids = list(id2node.keys())
    indeg2 = dict(indeg)

    # Выбор старта
    candidates = [nid for nid in node_ids if indeg2.get(nid, 0) == 0]
    if not candidates: candidates = list(node_ids)
    
    if orientation == "horizontal":
        start = min(candidates, key=lambda nid: bbox_center(id2node[nid]["bbox"])[0])
    else:
        start = min(candidates, key=lambda nid: bbox_center(id2node[nid]["bbox"])[1])

    ready = [nid for nid in node_ids if indeg2.get(nid, 0) == 0]
    if not ready and node_ids: ready = [start]

    order = []
    used = set()
    while ready:
        ready.sort(key=lambda nid: node_sort_key(nid, id2node, orientation))
        cur = ready.pop(0)
        if cur in used: continue
        used.add(cur)
        order.append(cur)
        for nxt in adj.get(cur, []):
            indeg2[nxt] -= 1
            if indeg2[nxt] <= 0 and nxt not in used:
                ready.append(nxt)

    leftovers = [nid for nid in node_ids if nid not in used]
    leftovers.sort(key=lambda nid: node_sort_key(nid, id2node, orientation))
    order.extend(leftovers)
    return order

# -----------------------------
# 3. Текст и Лейны
# -----------------------------

def match_text_to_nodes(node_items, full_ocr_items, iou_thr=0.05):
    node_text = {}
    # full_ocr_items - это результат ocr_object_result.json
    text_items = [it for it in full_ocr_items if it.get("text")]

    for nid, n in node_items:
        nb = n["bbox"]
        # 1. Попытка по IoU (для текста внутри блока)
        best = None
        best_iou = 0.0
        for it in text_items:
            # bbox в OCR может быть [x,y,w,h]
            iou = bbox_iou(nb, it["bbox"])
            if iou > best_iou:
                best_iou = iou
                best = it
        
        if best and best_iou >= iou_thr:
            node_text[nid] = best["text"].strip()
            continue
            
        # 2. Попытка по расстоянию (если текст рядом/чуть смещен)
        nc = bbox_center(nb)
        best = None
        best_d = float("inf")
        for it in text_items:
            tc = bbox_center(it["bbox"])
            d = dist(nc, tc)
            if d < best_d:
                best_d = d
                best = it
        
        # Если расстояние достаточно маленькое (например < 100 пикселей)
        if best and best_d < 100:
            node_text[nid] = best["text"].strip()
        else:
            node_text[nid] = ""
            
    return node_text

def dedupe_by_plane_y(items, orientation, eps=35):
    groups = defaultdict(list)
    for iid, obj in items:
        # Группируем по Y (или X если вертикальная ориентация, но обычно лейн-хедеры идут стопкой)
        # Упрощение: используем логику из вашего кода
        x, y, w, h = obj["bbox"]
        key = round(y / eps)
        groups[key].append((iid, obj))

    chosen = []
    for _, group in groups.items():
        # Берем самый левый/верхний из группы дублей
        if orientation == "horizontal":
            best = min(group, key=lambda it: it[1]["bbox"][0])
        else:
            best = min(group, key=lambda it: it[1]["bbox"][1])
        chosen.append(best)
    
    # Сортировка финального списка
    if orientation == "horizontal":
        chosen.sort(key=lambda it: it[1]["bbox"][0]) # На самом деле для горизонтальных лейнов важен Y
        # Исправляем логику сортировки дорожек: они идут сверху вниз
        chosen.sort(key=lambda it: it[1]["bbox"][1]) 
    else:
        chosen.sort(key=lambda it: it[1]["bbox"][0]) # Для вертикальных - слева направо

    return chosen

def assign_container_to_nodes(node_items, containers_sorted, orientation, margin=5.0):
    node2container = {}
    for nid, n in node_items:
        cx, cy = bbox_center(n["bbox"])
        best_id = None
        best_score = float("inf")

        for cid, cobj in containers_sorted:
            x, y, w, h = cobj["bbox"]
            right, bottom = x + w, y + h

            inside = (x <= cx <= right) and (y <= cy <= bottom)
            
            # Для горизонтальной диаграммы дорожки идут полосами по Y
            if orientation == "horizontal":
                in_band = (y <= cy <= bottom)
                belongs = inside or in_band
                score = abs(cy - (y + h / 2.0))
            else:
                in_band = (x <= cx <= right)
                belongs = inside or in_band
                score = abs(cx - (x + w / 2.0))

            if belongs and score < best_score:
                best_score = score
                best_id = cid
        node2container[nid] = best_id
    return node2container

def match_lane_labels_fixed(lanes_deduped, lane_label_items, orientation):
    """
    Сопоставляет OCR результаты (lane_label_items) с геометрическими лейнами (lanes_deduped).
    """
    lane_id2name = {}
    
    # lane_label_items это список dict: [{'lane_name': '...', 'bbox': ...}, ...]
    labels = [it for it in lane_label_items if it.get("lane_name")]

    for lane_id, lane_obj in lanes_deduped:
        lx, ly, lw, lh = lane_obj["bbox"]
        
        # Определяем зону поиска названия (заголовок обычно слева или сверху)
        best_label = None
        # Ищем лейбл, который пересекается с левой частью лейна (для horizontal)
        # или просто ближайший
        
        lane_center = (lx, ly + lh/2)
        min_dist = float('inf')
        
        for lbl in labels:
            lbl_center = bbox_center(lbl['bbox'])
            # Расстояние от центра лейбла до "начала" дорожки
            d = dist(lane_center, lbl_center)
            
            # Или проверка на IoU, если заголовок внутри
            iou = bbox_iou(lane_obj['bbox'], lbl['bbox'])
            
            if iou > 0 or d < min_dist:
                if d < min_dist:
                    min_dist = d
                    best_label = lbl['lane_name']
        
        if best_label and min_dist < 500: # Эвристика дальности
            lane_id2name[lane_id] = best_label
        else:
            lane_id2name[lane_id] = "Unnamed Lane"
            
    return lane_id2name

# -----------------------------
# 4. ГЛАВНАЯ ФУНКЦИЯ КОНВЕРТАЦИИ
# -----------------------------
def generate_bpmn_markdown(objects_data, arrows_data, ocr_objects_data, ocr_lanes_data):
    """
    Принимает списки словарей и возвращает BPMN процесс в виде Markdown-таблицы.
    """
    # 1. Подготовка (Логика остается прежней)
    node_items = filter_nodes_for_graph(objects_data)
    if not node_items:
        return "No process nodes found."

    orientation = detect_orientation(node_items)
    adj, indeg = build_graph(node_items, arrows_data)
    order = restore_order(node_items, adj, indeg, orientation)
    node_text = match_text_to_nodes(node_items, ocr_objects_data)

    # 2. Работа с дорожками (Логика остается прежней)
    raw_lanes = extract_by_class(objects_data, {"lane", "pool"})
    lanes_deduped = dedupe_by_plane_y(raw_lanes, orientation)
    lane_id2name = match_lane_labels_fixed(lanes_deduped, ocr_lanes_data, orientation)
    node2lane = assign_container_to_nodes(node_items, lanes_deduped, orientation)

    # 3. ГЕНЕРАЦИЯ ТАБЛИЦЫ MARKDOWN
    # Заголовок как на скриншоте
    md_lines = ["# Описание", ""]

    # Шапка таблицы
    md_lines.append("| | Наименование действия | Роль |")
    md_lines.append("| :--- | :--- | :--- |")

    id2node = {nid: n for nid, n in node_items}

    for i, nid in enumerate(order):
        node = id2node[nid]

        # Текст действия (из OCR)
        text = node_text.get(nid, "Unnamed").replace("\n", " ")

        # Если нужно добавить тип узла (Task/Event), можно раскомментировать:
        # cat = node.get("category_name", "Node")
        # action_display = f"**[{cat}]** {text}"
        action_display = text

        # Название роли (из Lane)
        lane_id = node2lane.get(nid)
        lane_name = lane_id2name.get(lane_id, "Jira")  # "Jira" как дефолт, если не распознано

        # Обработка переносов строк в роли (если там список через точку как на фото)
        # В Markdown таблицах перенос строки внутри ячейки делается через <br>
        role_display = lane_name.replace("\n", "<br>")

        # Формируем строку таблицы
        # На скриншоте первая колонка — это жирный номер
        md_lines.append(f"| **{i + 1}** | {action_display} | {role_display} |")

    return "\n".join(md_lines)
def main():
    st.set_page_config(page_title="BPMN to Markdown", layout="wide")
    st.title("Конвертер диаграмм в текстовый алгоритм в формате md")

    st.sidebar.header("Настройки распознавания")
    default_selection = ['task', 'event', 'exclusiveGateway', 'parallelGateway', 'lane']

    selected_labels = st.sidebar.multiselect(
        "Выберите элементы для поиска:",
        options=BPMN_CLASSES,
        default=[cls for cls in default_selection if cls in BPMN_CLASSES]
    )

    # Формируем итоговый список ID для передачи в модель
    target_class_ids = [CLASS_MAPPING[label] for label in selected_labels]

    # Загрузка моделей
    object_detection_model, arrow_detection_model = load_detection_models()
    ocr_object_model, ocr_lane_model = load_ocr_model()

    uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Загруженное изображение', width='stretch')

        if st.button("Запустить обработку"):
            progress_bar = st.progress(0)
            status = st.empty()

            try:
                # 1. DETECT OBJECTS
                status.text("Детекция узлов (Tasks, Events)...")
                object_result = get_sliced_prediction(
                    image_np, object_detection_model,
                    slice_height=640, slice_width=640,
                    overlap_height_ratio=0.3, overlap_width_ratio=0.3,
                    perform_standard_pred=True, postprocess_type='NMS', postprocess_match_threshold=0.3
                )

                # Фильтруем и чистим
                object_result.object_prediction_list = [
                    obj for obj in object_result.object_prediction_list
                    if obj.category.id in target_class_ids or obj.category.id == LANE_CLASS_ID
                ]
                object_result.object_prediction_list = clean_lane_overlaps(
                    object_result.object_prediction_list, LANE_CLASS_ID
                )
                # Получаем список словарей (COCO format)
                object_predictions_coco = object_result.to_coco_predictions()
                progress_bar.progress(25)

                # 2. OCR OBJECTS (Текст на узлах)
                status.text("OCR текста на узлах...")
                ocr_results_data = run_ocr_on_detections(
                    image_np, object_result.object_prediction_list, ocr_object_model, target_class_ids
                )
                progress_bar.progress(50)

                # 3. OCR LANES (Названия дорожек)
                status.text("Распознавание дорожек...")
                lane_results = get_lane_names_final(image_np, ocr_lane_model)
                progress_bar.progress(65)

                # 4. DETECT ARROWS
                status.text("Поиск связей (стрелок)...")
                arrow_result = get_sliced_prediction(
                    image_np, arrow_detection_model,
                    slice_height=640, slice_width=640,
                    overlap_height_ratio=0.4, overlap_width_ratio=0.4,
                    postprocess_type='NMM', postprocess_match_threshold=0.2, verbose=0
                )
                arrow_predictions_coco = arrow_result.to_coco_predictions()
                progress_bar.progress(85)

                # 5. ГЕНЕРАЦИЯ MARKDOWN
                status.text("Сборка алгоритма...")

                final_markdown = generate_bpmn_markdown(
                    object_predictions_coco, # Список словарей объектов
                    arrow_predictions_coco,  # Список словарей стрелок
                    ocr_results_data,        # OCR результаты (узлы)
                    lane_results             # OCR результаты (дорожки)
                )

                progress_bar.progress(100)
                status.success("Готово!")

                # Отображение и скачивание
                with col2:
                    st.subheader("Результат")
                    st.markdown(final_markdown)
                    st.download_button(
                        label="Скачать .md файл",
                        data=final_markdown,
                        file_name="process_algorithm.md",
                        mime="text/markdown"
                    )

            except Exception as e:
                st.error(f"Ошибка при обработке: {e}")
                import traceback
                st.code(traceback.format_exc())
    
if __name__ == "__main__":
    main()
