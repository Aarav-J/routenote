from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np
import os
from tempfile import NamedTemporaryFile
from ultralytics import YOLO
import colordetector as cd
from sklearn.cluster import AgglomerativeClustering
import json
from typing import List, Dict, Any
import time

OUTPUT_DIR = "static/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the YOLO model
model = YOLO("../../runs/detect/train9/weights/best.pt")

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image with the YOLO model and perform color clustering
    """
    start_time = time.time()

    # Save the uploaded file to a temporary location
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        contents = await file.read()
        temp_file.write(contents)
    
    try:
        # Read the image
        raw = cv2.imread(temp_file_path)
        if raw is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to process the uploaded image"}
            )

        # Analyze the image
        results = model.predict(raw, conf=0.5, iou=0.3, save=False, verbose=False)
        r = results[0]
        names = model.model.names

        # Extract detections with color information
        detections = []
        for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            crop = raw[y1:y2, x1:x2]
            
            # Make sure crop is valid
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            # Get color name and HSV values
            color_name, hsv_values = cd.classify_color(crop)
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf),
                "class_id": int(cls),
                "class_name": names[int(cls)],
                "color_name": color_name,
                "hsv": hsv_values.tolist() if isinstance(hsv_values, np.ndarray) else hsv_values,
            })

        # Perform color clustering
        if len(detections) > 0:
            hsv_values = np.array([d["hsv"] for d in detections])

            # Normalize HSV values for clustering
            normalized_hsv = np.zeros_like(hsv_values, dtype=float)
            normalized_hsv[:, 0] = hsv_values[:, 0] / 180.0
            normalized_hsv[:, 1] = hsv_values[:, 1] / 255.0
            normalized_hsv[:, 2] = hsv_values[:, 2] / 255.0

            # Weight hue more heavily
            normalized_hsv[:, 0] *= 2.0
            
            # Agglomerative clustering
            clustering = AgglomerativeClustering(
                distance_threshold=0.45,
                n_clusters=None,
                linkage='ward',
                metric='euclidean'
            )
            labels = clustering.fit_predict(normalized_hsv)

            # Add cluster labels to detections
            for d, label in zip(detections, labels):
                d["cluster"] = int(label)

            # Group detections by cluster
            groups = {}
            for d in detections:
                cluster_id = d["cluster"]
                if cluster_id not in groups:
                    groups[cluster_id] = []
                groups[cluster_id].append(d)

            # Generate cluster stats
            cluster_stats = []
            for cluster_id, items in groups.items():
                hsv_values = np.array([d["hsv"] for d in items])
                avg_h = int(np.mean(hsv_values[:, 0]))
                avg_s = int(np.mean(hsv_values[:, 1]))
                avg_v = int(np.mean(hsv_values[:, 2]))
                
                # Convert average HSV to RGB for frontend display
                rgb_color = cv2.cvtColor(
                    np.uint8([[[avg_h, avg_s, avg_v]]]),
                    cv2.COLOR_HSV2BGR
                )[0][0].tolist()
                
                cluster_stats.append({
                    "cluster_id": cluster_id,
                    "count": len(items),
                    "avg_hsv": [avg_h, avg_s, avg_v],
                    "rgb_color": rgb_color,
                    "items": items
                })

            # Sort clusters by size
            cluster_stats.sort(key=lambda x: x["count"], reverse=True)
        else:
            cluster_stats = []

        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

        processing_time = time.time() - start_time
        timestamp = int(time.time())
        filename = f"{timestamp}_annotated.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        img = raw.copy()
        for index, group in groups.items(): 
            color = (int((index * 30) % 180), 230, 230)
            bgr_color = tuple(reversed(cv2.cvtColor(
                np.array([[[color[0], color[1], color[2]]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR)[0][0].tolist()))
            for d in group:
                x1, y1, x2, y2 = map(int, d["bbox"])
                label = f'{index}'
                cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 2)
                cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
        cv2.imwrite(output_path, img)
        return {
            "success": True,
            "filename": file.filename,
            "total_detections": len(detections),
            "processing_time_seconds": processing_time,
            "num_clusters": len(cluster_stats),
            "clusters": cluster_stats
        }

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.get("/")
async def root():
    return {"message": "Rock climbing hold detection API. POST an image to /api/analyze"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)