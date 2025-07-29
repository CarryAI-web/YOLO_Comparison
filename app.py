import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# Translation dictionary for English and Chinese
translations = {
    "en": {
        "page_title": "YOLO Object Detection Model Comparison",
        "title": "YOLO Object Detection Model Comparison",
        "upload_prompt": "Upload an image and choose a detection task to compare object detection results.",
        "uploaded_image": "Uploaded Image",
        "detection_results": "Detection Results - {task_name} [Image Size] (All Models are developed in 100 epochs)",
        "results_caption": "Results from {model_name}",
        "detection_details": "Detection Details",
        "no_objects": "No objects detected.",
        "comparison_summary": "Comparison Summary - {task_name}",
        "no_image_prompt": "Press any the buttons to show detection results.",
        "task1_button": "Compare performance of our Trained Models",
        "task2_button": "Compare Default YOLO11 Model with Our Developed Model",
        "spinner": "Running object detection for {task_name}...",
        "Model": "Model",
        "Number of Detections": "Number of Detections",
        "Average Confidence": "Average Confidence",
        "class": "Class",
        "confidence": "Confidence",
    },
    "zh": {
        "page_title": "YOLO 物件偵察模型評測",
        "title": "YOLO 物件偵察模型評測",
        "upload_prompt": "請上載一張圖片以比較不同自主訓練模型的結果。",
        "uploaded_image": "已上載的圖片",
        "detection_results": "偵察結果 - {task_name} [圖片尺寸] (所有模型的訓練週期均為 100)",
        "results_caption": "{model_name} 的偵察結果",
        "detection_details": "偵察概要",
        "no_objects": "未偵察到任何物件。",
        "comparison_summary": "性能比較表 - {task_name}",
        "no_image_prompt": "點選以上按鈕以展示訓練結果。",
        "task1_button": "比較各個自行訓練模型的性能",
        "task2_button": "比較預設模型與自行訓練模型的性能",
        "spinner": "正在為 {task_name} 進行物件偵察...",
        "Model": "模型",
        "Number of Detections": "偵察數量",
        "Average Confidence": "平均信賴度",
        "class": "類別",
        "confidence": "信賴區間",
    }
}

# Initialize session state
if 'selected_task' not in st.session_state:
    st.session_state.selected_task = None
    st.session_state.results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # Default to English only if not set

col_left, col_right = st.columns([4, 1])
with col_right:
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        lan1 = st.button("English", key="lang_en")
    with col_btn2:
        lan2 = st.button("中文", key="lang_zh")

# Update language in session state
if lan1:
    st.session_state.language = 'en'
elif lan2:
    st.session_state.language = 'zh'

# Get current language translations
lang = st.session_state.language
t = translations[lang]

# Streamlit page configuration
st.set_page_config(page_title=t["page_title"], page_icon="🔍", layout="wide")
st.title(t["title"])
st.write(t["upload_prompt"])

# File uploader
uploaded_file = st.file_uploader("Choose an image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if st.session_state.language == "en":
    # FastAPI endpoint URLs
    MODEL_COMP_URL = {
        "YOLO11 Medium [890]": "http://127.0.0.1:8005/predict/model1/",
        "YOLO11 Medium [640]": "http://127.0.0.1:8005/predict/model2/",
        "YOLO11 Large [640]": "http://127.0.0.1:8005/predict/model3/",
        "Currently deployed model [640]": "http://127.0.0.1:8005/predict/model4/"
    }

    Task2_URL = {
        "Default YOLO11 Medium [640]": "http://127.0.0.1:8005/predict/model5/",
        "Currently deployed model [640]": "http://127.0.0.1:8005/predict/model4/",
        "YOLO11 Medium [890]": "http://127.0.0.1:8005/predict/model1/",
        "Pose Detection Model": "http://127.0.0.1:8005/predict/model6/"
    }

elif st.session_state.language == "zh":
    # FastAPI endpoint URLs
    MODEL_COMP_URL = {
        "YOLO11 Medium [890]": "http://127.0.0.1:8005/predict/model1/",
        "YOLO11 Medium [640]": "http://127.0.0.1:8005/predict/model2/",
        "YOLO11 Large [640]": "http://127.0.0.1:8005/predict/model3/",
        "現正使用中的模型 [640]": "http://127.0.0.1:8005/predict/model4/"
    }

    Task2_URL = {
        "YOLO11 默認模型 [640]": "http://127.0.0.1:8005/predict/model5/",
        "現正使用中的模型 [640]": "http://127.0.0.1:8005/predict/model4/",
        "YOLO11 Medium [890]": "http://127.0.0.1:8005/predict/model1/",
        "人體姿態模型": "http://127.0.0.1:8005/predict/model6/"
    }
# Function to process detection for a given set of URLs
def run_detection(file_bytes, urls, task_name):
    results = {}
    with st.spinner(t["spinner"].format(task_name=task_name)):
        for model_name, url in urls.items():
            try:
                # Prepare file for upload using in-memory bytes
                files = {"file": (uploaded_file.name, io.BytesIO(file_bytes), uploaded_file.type)}
                response = requests.post(url, files=files, timeout=30)

                if response.status_code == 200:
                    try:
                        results[model_name] = response.json()
                    except ValueError:
                        results[model_name] = {"error": "Invalid JSON response from server"}
                else:
                    results[model_name] = {"error": f"HTTP {response.status_code} error: {response.text}"}
            except requests.exceptions.RequestException as e:
                results[model_name] = {"error": f"Failed to connect to FastAPI server: {str(e)}"}
    
    return results, task_name


if st.session_state.language == "zh":
    # Function to display results
    def display_results(results, task_name):
        st.subheader(t["detection_results"].format(task_name=task_name))
        model_names = list(results.keys())
    
        # Use 2x2 grid for both Task 1 and Task 2
        num_cols = 2
        num_rows = (len(model_names) + num_cols - 1) // num_cols
    
        for row in range(num_rows):
            with st.container():
                cols = st.columns(num_cols)
                for col_idx, model_idx in enumerate(range(row * num_cols, min((row + 1) * num_cols, len(model_names)))):
                    model_name = model_names[model_idx]
                    result = results[model_name]
                    with cols[col_idx % num_cols]:
                        st.subheader(model_name)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            # Display labeled image
                            st.image(result["result_image"], caption=t["results_caption"].format(model_name=model_name), use_container_width=True)

                            # Display detection details
                            if result["detections"]:
                                st.write(t["detection_details"])
                                df = pd.DataFrame(
                                    result["detections"],
                                    columns=["x1", "y1", "x2", "y2", "信賴度", "類別"]
                                )
                                df["類別"] = df["類別"].astype(int)
                                df["信賴度"] = df["信賴度"].round(2)
                                st.dataframe(df[["類別", "信賴度", "x1", "y1", "x2", "y2"]])
                            else:
                                st.write(t["no_objects"])

    # Summary metrics
        st.subheader(t["comparison_summary"].format(task_name=task_name))
        summary = {
            "模型": [],
            "偵察數量": [],
            "平均信賴度": []
        }
        for model_name, result in results.items():
            if "error" not in result and result["detections"]:
                summary["模型"].append(model_name)
                summary["偵察數量"].append(len(result["detections"]))
                confidences = [d[4] for d in result["detections"]]
                summary["平均信賴度"].append(round(sum(confidences) / len(confidences), 2) if confidences else 0.0)
            else:
                summary["模型"].append(model_name)
                summary["偵察數量"].append(0)
                summary["平均信賴度"].append(0.0)
        st.dataframe(pd.DataFrame(summary))

elif st.session_state.language == "en":
    def display_results(results, task_name):
        st.subheader(t["detection_results"].format(task_name=task_name))
        model_names = list(results.keys())
    
        # Use 2x2 grid for both Task 1 and Task 2
        num_cols = 2
        num_rows = (len(model_names) + num_cols - 1) // num_cols
    
        for row in range(num_rows):
            with st.container():
                cols = st.columns(num_cols)
                for col_idx, model_idx in enumerate(range(row * num_cols, min((row + 1) * num_cols, len(model_names)))):
                    model_name = model_names[model_idx]
                    result = results[model_name]
                    with cols[col_idx % num_cols]:
                        st.subheader(model_name)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            # Display labeled image
                            st.image(result["result_image"], caption=t["results_caption"].format(model_name=model_name), use_container_width=True)

                            # Display detection details
                            if result["detections"]:
                                st.write(t["detection_details"])
                                df = pd.DataFrame(
                                    result["detections"],
                                    columns=["x1", "y1", "x2", "y2", "confidence", "class"]
                                )
                                df["class"] = df["class"].astype(int)
                                df["confidence"] = df["confidence"].round(2)
                                st.dataframe(df[["class", "confidence", "x1", "y1", "x2", "y2"]])
                            else:
                                st.write(t["no_objects"])

    # Summary metrics
        st.subheader(t["comparison_summary"].format(task_name=task_name))
        summary = {
            "Model": [],
            "Number of Detections": [],
            "Average Confidence": []
        }
        for model_name, result in results.items():
            if "error" not in result and result["detections"]:
                summary["Model"].append(model_name)
                summary["Number of Detections"].append(len(result["detections"]))
                confidences = [d[4] for d in result["detections"]]
                summary["Average Confidence"].append(round(sum(confidences) / len(confidences), 2) if confidences else 0.0)
            else:
                summary["Model"].append(model_name)
                summary["Number of Detections"].append(0)
                summary["Average Confidence"].append(0.0)
        st.dataframe(pd.DataFrame(summary))

if uploaded_file is not None:
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.results = None
        st.session_state.selected_task = None
    # Read uploaded file into memory
    file_bytes = uploaded_file.read()

    # Display uploaded image
    st.subheader(t["uploaded_image"])
    st.image(file_bytes, use_container_width=True)

    # Create two buttons for different detection tasks
    col1, col2 = st.columns(2)
    with col1:
        task1_button = st.button(t["task1_button"], key="task1")
    with col2:
        task2_button = st.button(t["task2_button"], key="task2")

    # Execute detection based on button pressed
    if task1_button:
        st.session_state.selected_task = t["task1_button"]
        st.session_state.results = run_detection(file_bytes, MODEL_COMP_URL, t["task1_button"])
        display_results(*st.session_state.results)
    elif task2_button:
        st.session_state.selected_task = t["task2_button"]
        st.session_state.results = run_detection(file_bytes, Task2_URL, t["task2_button"])
        display_results(*st.session_state.results)
    elif st.session_state.selected_task and st.session_state.results:
        # Redisplay results if page is rerun but no new button is pressed
        display_results(*st.session_state.results)
else:
    st.session_state.uploaded_file = None
    st.session_state.results = None
    st.session_state.selected_task = None
    st.info(t["no_image_prompt"])

