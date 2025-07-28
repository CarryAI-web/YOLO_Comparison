"""import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="YOLO Model Comparison", page_icon="🔍", layout="wide")
st.title("YOLO Model Comparison")
st.write("Upload an image to compare object detection results from four YOLO models.")

# File uploader
uploaded_file = st.file_uploader("Please upload an image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

# FastAPI endpoint URLs
FASTAPI_URLS = {
    "Deployed Model": "http://127.0.0.1:8000/predict/model1/",
    "Retrained YOLO11S": "http://127.0.0.1:8000/predict/model2/",
    "Retrained YOLO11M": "http://127.0.0.1:8000/predict/model3/",
    "Retrained YOLO11L": "http://127.0.0.1:8000/predict/model4/"
}

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    st.image(uploaded_file, use_column_width=True)

    # Send image to all four FastAPI endpoints
    st.subheader("Detection Results")
    cols = st.columns(4)  # Create four columns for side-by-side display
    results = {}

    with st.spinner("Running object detection on all models..."):
        for model_name, url in FASTAPI_URLS.items():
            try:
                # Prepare file for upload
                uploaded_file.seek(0)  # Reset file pointer
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(url, files=files)

                if response.status_code == 200:
                    results[model_name] = response.json()
                else:
                    results[model_name] = {"error": response.json().get("detail", "Unknown error")}
            except requests.exceptions.RequestException as e:
                results[model_name] = {"error": f"Failed to connect to FastAPI server: {str(e)}"}
    
    # Display results in a 2x2 grid (2 columns per row, 2 rows)
    model_names = list(results.keys())
    for row in range(2):
        with st.container():
            cols = st.columns(2)  # Two columns per row
            for col_idx, model_idx in enumerate([row * 2, row * 2 + 1]):
                if model_idx < len(model_names):
                    model_name = model_names[model_idx]
                    result = results[model_name]
                    with cols[col_idx]:
                        st.subheader(model_name)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            # Display labeled image
                            st.image(result["result_image"], caption=f"Results from {model_name}", use_column_width=True)

                            # Display detection details
                            if result["detections"]:
                                st.write("Detection Details")
                                df = pd.DataFrame(
                                    result["detections"],
                                    columns=["x1", "y1", "x2", "y2", "confidence", "class"]
                                )
                                df["class"] = df["class"].astype(int)
                                df["confidence"] = df["confidence"].round(2)
                                st.dataframe(df[["class", "confidence", "x1", "y1", "x2", "y2"]])
                            else:
                                st.write("No objects detected.")

    # Summary metrics
    st.subheader("Comparison Summary")
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

import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="YOLO Model Comparison", page_icon="🔍", layout="wide")
st.title("YOLO Model Comparison")
st.write("Upload an image to compare object detection results from four YOLO models.")

# File uploader
uploaded_file = st.file_uploader("Choose an image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

# FastAPI endpoint URLs
MODEL_COMP_URL = {
    "yolo11 Medium [890]": "http://127.0.0.1:8005/predict/model1/",
    "yolo11 Medium [640]": "http://127.0.0.1:8005/predict/model2/",
    "yolo11 Large [640]": "http://127.0.0.1:8005/predict/model3/",
    "Currently deployed model [640]": "http://127.0.0.1:8005/predict/model4/"
}

Task2_URL = {
    "Default YOLO11 Medium [640]": "http://127.0.0.1:8005/predict/model6/",
    "Currently deploying model [640]": "http://127.0.0.1:8005/predict/model4/",
    "yolo11 Medium [890]": "http://127.0.0.1:8005/predict/model1/"
}


#if uploaded_file is not None:
#    # Read uploaded file into memory to avoid multiple reads
#    file_bytes = uploaded_file.read()

    # Display uploaded image
#    st.subheader("Uploaded Image")
#   st.image(file_bytes, use_container_width=True)

    # Send image to all four FastAPI endpoints
#    st.subheader("Detection Results [Image Size] (All Models are developed in 100 epochs)")
#    results = {}


# Function to process detection for a given set of URLs
def run_detection(file_bytes, urls, task_name):
    st.subheader(f"Detection Results - {task_name} [Image Size] (All Models are developed in 100 epochs)")
    results = {}
    with st.spinner("Running object detection on all models..."):
        for model_name, url in MODEL_COMP_URL.items():
            try:
                # Prepare file for upload using in-memory bytes
                files = {"file": (uploaded_file.name, io.BytesIO(file_bytes), uploaded_file.type)}
                response = requests.post(url, files=files, timeout=30)

                if response.status_code == 200:
                    results[model_name] = response.json()
                else:
                    results[model_name] = {"error": response.json().get("detail", f"HTTP {response.status_code} error")}
            except requests.exceptions.RequestException as e:
                results[model_name] = {"error": f"Failed to connect to FastAPI server: {str(e)}"}

def display_results(results, task_name):
    st.subheader(f"Detection Results - {task_name} [Image Size] (All Models are developed in 100 epochs)")
    model_names = list(results.keys())
    
    # Adjust grid for Task 2 (3 models: 2 in first row, 1 in second row)
    num_cols = 2
    num_rows = (len(model_names) + 1) // 2
    for row in range(num_rows):
        with st.container():
            cols = st.columns(2)  # Two columns per row
            for col_idx, model_idx in enumerate([row * 2, row * 2 + 1]):
                if model_idx < len(model_names):
                    model_name = model_names[model_idx]
                    result = results[model_name]
                    with cols[col_idx]:
                        st.subheader(model_name)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            # Display labeled image
                            st.image(result["result_image"], caption=f"Results from {model_name}", use_column_width=True)

                            # Display detection details
                            if result["detections"]:
                                st.write("Detection Details")
                                df = pd.DataFrame(
                                    result["detections"],
                                    columns=["x1", "y1", "x2", "y2", "confidence", "class"]
                                )
                                df["class"] = df["class"].astype(int)
                                df["confidence"] = df["confidence"].round(2)
                                st.dataframe(df[["class", "confidence", "x1", "y1", "x2", "y2"]])
                            else:
                                st.write("No objects detected.")


    # Summary metrics
    st.subheader("Comparison Summary")
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
    # Read uploaded file into memory
    file_bytes = uploaded_file.read()

    # Display uploaded image
    st.subheader("Uploaded Image")
    st.image(file_bytes, use_container_width=True)

    # Create two buttons for different detection tasks
    col1, col2 = st.columns(2)
    with col1:
        task1_button = st.button("Compare performance of our Trained Models")
    with col2:
        task2_button = st.button("Compare Default YOLO11 Model with Our Model")

    # Execute detection based on button pressed
    if task1_button:
        run_detection(file_bytes, MODEL_COMP_URL, "Compare performance of our Trained Models")
    if task2_button:
        run_detection(file_bytes, Task2_URL, "Compare Default YOLO11 Model with Our Model")

    if task1_button:
        st.session_state.selected_task = "Compare performance of our Trained Models"
        results, task_name = run_detection(file_bytes, MODEL_COMP_URL, "Compare performance of our Trained Models")
        display_results(results, task_name)
    elif task2_button:
        st.session_state.selected_task = "Compare Default YOLO11 Model with Our Model"
        results, task_name = run_detection(file_bytes, Task2_URL, "Compare Default YOLO11 Model with Our Model")
        display_results(results, task_name)
    elif st.session_state.selected_task:
        # Redisplay results if page is rerun but no new button is pressed g
        if st.session_state.results:
            results, task_name = st.session_state.results
            display_results(results, task_name)


            




import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="YOLO Model Comparison", page_icon="🔍", layout="wide")
st.title("YOLO Model Comparison")
st.write("Upload an image and choose a detection task to compare object detection results.")

# Initialize session state
if 'selected_task' not in st.session_state:
    st.session_state.selected_task = None
    st.session_state.results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# File uploader
uploaded_file = st.file_uploader("Choose an image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

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

# Function to process detection for a given set of URLs
def run_detection(file_bytes, urls, task_name):
    results = {}
    with st.spinner(f"Running object detection for {task_name}..."):
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

# Function to display results
def display_results(results, task_name):
    st.subheader(f"Detection Results - {task_name} [Image Size] (All Models are developed in 100 epochs)")
    model_names = list(results.keys())
    
    # Use 2x2 grid for Task 1 (4 models), 3x1 grid for Task 2 (3 models)
    num_cols = 2 if task_name == "Compare performance of our Trained Models" else 1
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
                        st.image(result["result_image"], caption=f"Results from {model_name}", use_container_width=True)

                        # Display detection details
                        if result["detections"]:
                            st.write("Detection Details")
                            df = pd.DataFrame(
                                result["detections"],
                                columns=["x1", "y1", "x2", "y2", "confidence", "class"]
                            )
                            df["class"] = df["class"].astype(int)
                            df["confidence"] = df["confidence"].round(2)
                            st.dataframe(df[["class", "confidence", "x1", "y1", "x2", "y2"]])
                        else:
                            st.write("No objects detected.")

    # Summary metrics
    st.subheader(f"Comparison Summary - {task_name}")
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
    st.subheader("Uploaded Image")
    st.image(file_bytes, use_container_width=True)

    # Create two buttons for different detection tasks
    col1, col2 = st.columns(2)
    with col1:
        task1_button = st.button("Compare performance of our Trained Models")
    with col2:
        task2_button = st.button("Compare Default YOLO11 Model with Our Model")

    # Execute detection based on button pressed
    if task1_button:
        st.session_state.selected_task = "Compare performance of our Trained Models"
        st.session_state.results = run_detection(file_bytes, MODEL_COMP_URL, "Compare performance of our Trained Models")
        display_results(*st.session_state.results)
    elif task2_button:
        st.session_state.selected_task = "Compare Default YOLO11 Model with Our Model"
        st.session_state.results = run_detection(file_bytes, Task2_URL, "Compare Default YOLO11 Model with Our Model")
        display_results(*st.session_state.results)
    elif st.session_state.selected_task and st.session_state.results:
        # Redisplay results if page is rerun but no new button is pressed
        display_results(*st.session_state.results)
    else:
        st.session_state.uploaded_file = None
        st.session_state.results = None
        st.session_state.selected_task = None
        st.info("You can press one of the button to reveal the detection results.") 





import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# Streamlit page configuration
col_left, col_right = st.columns([4, 1])
with col_right: 
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        lan1 = st.button("English")
    with col_btn2:
        lan2 = st.button("中文")

if (lan1 == False and lan2 == False) or lan1 == True:
    st.set_page_config(page_title="YOLO Model Comparison", page_icon="🔍", layout="wide")
    st.title("YOLO Model Comparison")
    st.write("Upload an image and choose a detection task to compare object detection results.")
elif lan2 == True:
    st.set_page_config(page_title="YOLO 模型評測", page_icon="🔍", layout="wide")
    st.title("YOLO 模型評測")
    st.write("請上載一張圖片以進行不同 YOLO 模型比較")

# Initialize session state
if 'selected_task' not in st.session_state:
    st.session_state.selected_task = None
    st.session_state.results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

def language_select ():
    if (lan1 == False and lan2 == False) or lan1 == True:
        return "en"
    elif lan2 == True:
        return "zh"


# File uploader
uploaded_file = st.file_uploader("Choose an image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

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

# Function to process detection for a given set of URLs
def run_detection(file_bytes, urls, task_name):
    results = {}
    with st.spinner(f"Running object detection for {task_name}..."):
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

# Function to display results
def display_results(results, task_name):
    st.subheader(f"Detection Results - {task_name} [Image Size] (All Models are developed in 100 epochs)")
    model_names = list(results.keys())
    
    # Use 2x2 grid for Task 1 (4 models), 3x1 grid for Task 2 (3 models)
    num_cols = 2 if task_name == "Compare performance of our Trained Models" else 1
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
                        st.image(result["result_image"], caption=f"Results from {model_name}", use_container_width=True)

                        # Display detection details
                        if result["detections"]:
                            st.write("Detection Details")
                            df = pd.DataFrame(
                                result["detections"],
                                columns=["x1", "y1", "x2", "y2", "confidence", "class"]
                            )
                            df["class"] = df["class"].astype(int)
                            df["confidence"] = df["confidence"].round(2)
                            st.dataframe(df[["class", "confidence", "x1", "y1", "x2", "y2"]])
                        else:
                            st.write("No objects detected.")

    # Summary metrics
    st.subheader(f"Comparison Summary - {task_name}")
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
    st.subheader("Uploaded Image")
    st.image(file_bytes, use_container_width=True)

    # Create two buttons for different detection tasks
    col1, col2 = st.columns(2)
    with col1:
        task1_button = st.button("Compare performance of our Trained Models")
    with col2:
        task2_button = st.button("Compare Default YOLO11 Model with Our Model")

    # Execute detection based on button pressed
    if task1_button:
        st.session_state.selected_task = "Compare performance of our Trained Models"
        st.session_state.results = run_detection(file_bytes, MODEL_COMP_URL, "Compare performance of our Trained Models")
        display_results(*st.session_state.results)
    elif task2_button:
        st.session_state.selected_task = "Compare Default YOLO11 Model with Our Model"
        st.session_state.results = run_detection(file_bytes, Task2_URL, "Compare Default YOLO11 Model with Our Model")
        display_results(*st.session_state.results)
    elif st.session_state.selected_task and st.session_state.results:
        # Redisplay results if page is rerun but no new button is pressed
        display_results(*st.session_state.results)
    else:
        st.session_state.uploaded_file = None
        st.session_state.results = None
        st.session_state.selected_task = None
        st.info("You can press one of the button to reveal the detection results.")"""




import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

# Translation dictionary for English and Chinese
translations = {
    "en": {
        "page_title": "YOLO Model Comparison",
        "title": "YOLO Model Comparison",
        "upload_prompt": "Upload an image and choose a detection task to compare object detection results.",
        "uploaded_image": "Uploaded Image",
        "detection_results": "Detection Results - {task_name} [Image Size] (All Models are developed in 100 epochs)",
        "results_caption": "Results from {model_name}",
        "detection_details": "Detection Details",
        "no_objects": "No objects detected.",
        "comparison_summary": "Comparison Summary - {task_name}",
        "no_image_prompt": "You can press one of the buttons to reveal the detection results.",
        "task1_button": "Compare performance of our Trained Models",
        "task2_button": "Compare Default YOLO11 Model with Our Model",
        "spinner": "Running object detection for {task_name}..."
    },
    "zh": {
        "page_title": "YOLO 模型評測",
        "title": "YOLO 模型評測",
        "upload_prompt": "請上載一張圖片並選擇檢測任務以比較不同 YOLO 模型的結果。",
        "uploaded_image": "已上載的圖片",
        "detection_results": "檢測結果 - {task_name} [圖片尺寸] (所有模型的訓練週期均為 100)",
        "results_caption": "{model_name} 的檢測結果",
        "detection_details": "檢測詳情",
        "no_objects": "未檢測到任何物件。",
        "comparison_summary": "比較結果 - {task_name}",
        "no_image_prompt": "您可以點選其中一個按鈕以顯示檢測結果。",
        "task1_button": "比較我們訓練模型的性能",
        "task2_button": "比較預設 YOLO11 模型與我們的模型",
        "spinner": "正在為 {task_name} 進行物體檢測..."
    }
}

# Initialize session state
if 'selected_task' not in st.session_state:
    st.session_state.selected_task = None
    st.session_state.results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # Default to English

# Language selection buttons
col_left, col_right = st.columns([4, 1])
with col_right:
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        lan1 = st.button("English", key="lang_en")
    with col_btn2:
        lan2 = st.button("中文", key="lang_zh")

# Update language in session state
if (lan1 == False and lan2 == False) or lan1 == True:
    st.session_state.language = 'en'
elif lan2 == True:
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

# Function to display results
def display_results(results, task_name):
    st.subheader(t["detection_results"].format(task_name=task_name))
    model_names = list(results.keys())
    
    # Use 2x2 grid for Task 1 (4 models), 3x1 grid for Task 2 (3 models)
    num_cols = 2 if task_name == t["task1_button"] else 1
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
    col1, col2 = st.columns([4, 1])
#with col_right:
##col_btn1, col_btn2 = st.columns(2)
##with col_btn1:
###task1_button = st.button(t["task1_button"], key="task1")
##with col_btn2:
###task2_button = st.button(t["task2_button"], key="task2")
##col1, col2 = st.columns(2)
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