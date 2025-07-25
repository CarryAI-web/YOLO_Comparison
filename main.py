"""                This commented code is for ONE MODEL ANALYSIS ONLY
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel 
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np



class ImageRequest(BaseModel):
    image: Union[str, bytes]  # Accepts either a file path or raw image bytes


app = FastAPI()
YOLO_MODEL_PATH = "yolo11m_2025-06-25.pt"  # Path to your YOLO model file
model = YOLO(YOLO_MODEL_PATH)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg, .jpeg, and .png files are allowed.")
    
    # Read the image file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the image.")

    # Perform inference
    results = model(image)
    rst_img = results[0].plot()
    # Process results (for simplicity, returning the first result)
    if results:
        # return {"detections": results[0].boxes.xyxy.cpu().numpy().tolist()}
        return rst_img.tolist()  # Convert the image with detections to a list format for JSON response
    else:
        return {"detections": []}
@app.get("/")
async def root():
    return {"message": "Welcome to the YOLOv11 Detection API. Use the /predict endpoint to upload an image for detection."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, host="127.0.0.1")"""




"""     This commented code is for THREE MODEL ANALYSIS, with different code structure
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from pydantic import BaseModel

app = FastAPI(
    title="YOLOv11 Detection API",
    description="API for object detection using YOLOv11 model",
    version="1.0.0"
)
templates = Jinja2Templates(directory="templates")

# Path to your YOLO model file
YOLO_MODEL_PATH1 = "/Users/continental/Desktop/Intern/pt models/yolo11m_2025-06-25.pt"
YOLO_MODEL_PATH2 =
YOLO_MODEL_PATH3 =

# Load YOLOv8 model at startup
@app.on_event("startup")
async def load_model():
    global model
    model1 = YOLO(YOLO_MODEL_PATH1)

# Response model for detection results
class DetectionResult(BaseModel):
    filename: str
    result_image: str
    detections: list

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict/", response_model=DetectionResult)
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg, .jpeg, and .png files are allowed.")

    # Read the image file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the image.")

    # Perform inference
    results = model(image)
    rst_img = results[0].plot()  # Get labeled image with bounding boxes and labels

    # Convert BGR to RGB for correct display
    rst_img_rgb = cv2.cvtColor(rst_img, cv2.COLOR_BGR2RGB)

    # Convert to base64 for JSON response
    _, buffer = cv2.imencode('.jpg', rst_img_rgb)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Extract detection details (optional)
    detections = results[0].boxes.data.cpu().numpy().tolist()  # [x1, y1, x2, y2, conf, class]

    return {
        "filename": file.filename,
        "result_image": f"data:image/jpeg;base64,{img_base64}",
        "detections": detections
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, host="127.0.0.1")"""




from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from pydantic import BaseModel
import os

app = FastAPI(
    title="YOLO Model Comparison API",
    description="API for object detection using multiple YOLO models",
    version="1.0.0"
)
templates = Jinja2Templates(directory="templates")

# Add CORS to allow requests from Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://carryai-yolo-test.streamlit.app", "http://localhost:8501"],  # Adjust for your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolo11m.pt")
model = YOLO("classes.txt").load("yolo11m.pt")  # Default YOLOv11 Medium model, 640x640
model2 = YOLO("yolo11m-pose.pt")
model2 = YOLO("classes.txt").load("yolo11m-pose.pt")

# Paths to your YOLO model files
MODEL_PATHS = {
    "model1": "yolo11m_890_100e.pt", #890x890 with best performance
    "model2": "yolo11m_640_100e.pt", #640x640 with same structure as model 1
    "model3": "yolo11l_640_100e.pt", #YOLO11 Large model, 640x640 YOLO11 Medium model, 640x640
    "model4": "yolov11m_640_old.pt",  # YOLO11 Medium model, 640x640, old version
    "model5": "yolo11s_1280_100e.pt", # YOLO11 Small model, 1280x1280
    "model6": "yolo11m_1ep", # Default YOLO11 Medium model, 640x640
    "model7": model2
}

# Load YOLO models at startup
@app.on_event("startup")
async def load_models():
    global models
    models = {
        "model1": YOLO(MODEL_PATHS["model1"]),
        "model2": YOLO(MODEL_PATHS["model2"]),
        "model3": YOLO(MODEL_PATHS["model3"]),
        "model4": YOLO(MODEL_PATHS["model4"]),
        "model5": YOLO(MODEL_PATHS["model5"]),
        "model6": YOLO(MODEL_PATHS["model6"]),
        "model7": YOLO(MODEL_PATHS["model7"])
    }

# Response model for detection results
class DetectionResult(BaseModel):
    filename: str
    result_image: str
    detections: list
    model_name: str

# Generic prediction function
async def predict_with_model(file: UploadFile, model_name: str):
    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg, .jpeg, and .png files are allowed.")

    # Read the image file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the image.")

    # Perform inference
    model = models[model_name]
    results = model(image)
    rst_img = results[0].plot()  # Get labeled image with bounding boxes and labels

    # Convert BGR to RGB for correct display
    # rst_img_rgb = cv2.cvtColor(rst_img, cv2.COLOR_BGR2RGB)
    rst_img_rgb = rst_img

    # Convert to base64 for JSON response
    _, buffer = cv2.imencode('.jpg', rst_img_rgb)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Extract detection details
    detections = results[0].boxes.data.cpu().numpy().tolist()  # [x1, y1, x2, y2, conf, class]

    return {
        "filename": file.filename,
        "result_image": f"data:image/jpeg;base64,{img_base64}",
        "detections": detections,
        "model_name": model_name
    }

# Endpoints for each model
@app.post("/predict/model1/", response_model=DetectionResult)
async def predict_model1(file: UploadFile = File(...)):
    return await predict_with_model(file, "model1")

@app.post("/predict/model2/", response_model=DetectionResult)
async def predict_model2(file: UploadFile = File(...)):
    return await predict_with_model(file, "model2")

@app.post("/predict/model3/", response_model=DetectionResult)
async def predict_model3(file: UploadFile = File(...)):
    return await predict_with_model(file, "model3")

@app.post("/predict/model4/", response_model=DetectionResult)
async def predict_model4(file: UploadFile = File(...)):
    return await predict_with_model(file, "model4")

@app.post("/predict/model5/", response_model=DetectionResult)
async def predict_model5(file: UploadFile = File(...)):
    return await predict_with_model(file, "model5") 

@app.post("/predict/model6/", response_model=DetectionResult)
async def predict_model6(file: UploadFile = File(...)):     
    return await predict_with_model(file, "model6")

@app.post("/predict/model7/", response_model=DetectionResult)
async def predict_model7(file: UploadFile = File(...)):
    return await predict_with_model(file, "model7")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8005, host="127.0.0.1")
