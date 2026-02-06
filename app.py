import time
import requests
import streamlit as st
from PIL import Image

API = "http://127.0.0.1:8000"

def main():
    st.set_page_config(page_title="BPMN to Markdown", layout="wide")
    st.title("Конвертер диаграмм в текстовый алгоритм в формате md")

    if "jobs" not in st.session_state:
        st.session_state.jobs = []  # list[dict]: {job_id, filename}

    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption="Загруженное изображение", width="stretch")

        if st.button("Отправить в очередь"):
            resp = requests.post(
                f"{API}/submit",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                data={"labels": ""}
            )
            data = resp.json()
            if resp.status_code != 200:
                st.error(data.get("error", "unknown error"))
            else:
                job_id = data["job_id"]
                st.session_state.jobs.insert(0, {"job_id": job_id, "filename": uploaded_file.name})
                st.success(f"Задача поставлена в очередь: {job_id}")

    # ---- Список задач ----
    st.divider()
    st.subheader("Очередь / задачи")

    if not st.session_state.jobs:
        st.info("Пока нет задач. Загрузи картинку и нажми «Отправить в очередь».")
        return

    # общий авто-рефреш по кнопке
    if st.button("Обновить статусы"):
        st.rerun()

    for j in st.session_state.jobs:
        job_id = j["job_id"]
        filename = j["filename"]

        with st.expander(f"{filename} — {job_id}", expanded=False):
            s = requests.get(f"{API}/status/{job_id}").json()
            st.json(s)

            status = s.get("status")
            if status == "done":
                if st.button(f"Показать результат ({job_id})", key=f"res_{job_id}"):
                    r = requests.get(f"{API}/result/{job_id}").json()
                    with col2:
                        st.subheader(f"Результат: {filename}")
                        st.markdown(r.get("markdown", ""))
                        st.download_button(
                            "Скачать .md файл",
                            r.get("markdown", ""),
                            f"{filename}.md",
                            mime="text/markdown"
                        )

            elif status == "error":
                st.error(s.get("error", "processing error"))

            else:
                st.info("В очереди / выполняется. Нажми «Обновить статусы» через пару секунд.")

if __name__ == "__main__":
    main()
