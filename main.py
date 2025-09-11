from ultralytics import YOLO
import cv2, json 
import colordetector as cd 
from sklearn.cluster import AgglomerativeClustering
import preprocess 
import numpy as np
model = YOLO("../../runs/detect/train9/weights/best.pt")
img_path = "./corec/4.png"
raw = cv2.imread(img_path)
# img = preprocess.preprocess_wall(raw)
img = raw

results = model.predict(img, conf=0.5, iou=0.3, save=False, verbose=False)

r = results[0]                       
names = model.model.names          


boxes = r.boxes.xyxy.cpu().numpy()   
confs = r.boxes.conf.cpu().numpy()
clses = r.boxes.cls.cpu().numpy() 


detections = []
# for (x1,y1,x2,y2), conf, cls in zip(boxes, confs, clses):
    
#     detections.append({
#         "bbox":[float(x1), float(y1), float(x2), float(y2)],
#         "conf": float(conf),
#         "class_id": int(cls),
#         "class_name": names[int(cls)]
#     })


for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
    x1,y1,x2,y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
    
    # Make sure crop is valid (some bounding boxes might be at the edge)
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        continue
        
    # Get color name and HSV values using our simplified approach
    color_name, hsv_values = cd.classify_color(crop)
    detections.append({
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "conf": float(conf),
        "class_id": int(cls),
        "class_name": names[int(cls)],
        "color_name": color_name,
        "hsv": hsv_values,
    })

# Extract HSV values for clustering
if len(detections) > 0:
    hsv_values = np.array([d["hsv"] for d in detections])

    # Normalize HSV values for clustering
    # H is in [0,180], S and V are in [0,255] in OpenCV
    normalized_hsv = np.zeros_like(hsv_values, dtype=float)
    normalized_hsv[:, 0] = hsv_values[:, 0] / 180.0
    normalized_hsv[:, 1] = hsv_values[:, 1] / 255.0
    normalized_hsv[:, 2] = hsv_values[:, 2] / 255.0

    # Weight hue more heavily than saturation and value
    normalized_hsv[:, 0] *= 2.0
    
    print(json.dumps(detections[:3], indent=2))
    
    # Agglomerative clustering with distance_threshold
    clustering = AgglomerativeClustering(
        distance_threshold=0.45,  
        n_clusters=None,
        linkage='ward',  
        metric='euclidean'
    )
    labels = clustering.fit_predict(normalized_hsv)
else:
    labels = []

for d, label in zip(detections, labels):
    d["cluster"] = int(label)

# Group detections by cluster
groups = {}
for d in detections:
    groups.setdefault(d["cluster"], []).append(d)

# Generate distinct colors for visualization
cluster_colors = {}
for index in groups.keys():
    # Use HSV to generate visually distinct colors
    hue = (index * 30) % 180  # Space hues evenly
    cluster_colors[index] = tuple(reversed(cv2.cvtColor(
        np.array([[[hue, 230, 230]]], dtype=np.uint8),
        cv2.COLOR_HSV2BGR)[0][0].tolist()))

# First, draw each cluster with its own color
original_img = raw.copy()
for index, group in groups.items():
    color = cluster_colors[index]
    for d in group:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f'{index}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Create an enhanced legend with actual colors, HSV values, and matching bounding box colors
def create_enhanced_legend(groups, cluster_colors, img_width=300):
    # Sort clusters by size (largest first) for the legend
    sorted_clusters = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Calculate height based on number of clusters
    row_height = 50
    padding = 10
    legend_height = len(sorted_clusters) * row_height + 2 * padding
    legend_img = np.ones((legend_height, img_width, 3), dtype=np.uint8) * 255
    
    y_pos = padding
    for i, (index, group) in enumerate(sorted_clusters):
        # Get the bounding box color used in visualization
        bbox_color = cluster_colors[index]
        
        # Calculate average HSV for this cluster
        hsv_values = np.array([d["hsv"] for d in group])
        avg_h = int(np.mean(hsv_values[:, 0]))
        avg_s = int(np.mean(hsv_values[:, 1]))
        avg_v = int(np.mean(hsv_values[:, 2]))
        
        # Create a rectangle with the actual average color
        actual_color = cv2.cvtColor(
            np.uint8([[[avg_h, avg_s, avg_v]]]),
            cv2.COLOR_HSV2BGR
        )[0][0].astype(int)
        
        # Draw the sample color swatch (actual color from HSV)
        cv2.rectangle(
            legend_img,
            (10, y_pos),
            (60, y_pos + 30),
            (int(actual_color[0]), int(actual_color[1]), int(actual_color[2])),
            -1
        )
        
        # Draw the bounding box color swatch (matches what's shown on image)
        cv2.rectangle(
            legend_img,
            (70, y_pos),
            (100, y_pos + 30),
            bbox_color,
            -1
        )
        
        # Add border to swatches for clarity
        cv2.rectangle(legend_img, (10, y_pos), (60, y_pos + 30), (0, 0, 0), 1)
        cv2.rectangle(legend_img, (70, y_pos), (100, y_pos + 30), (0, 0, 0), 1)
        
        # Add text information
        text = f"Cluster {index}: {len(group)} holds"
        cv2.putText(
            legend_img,
            text,
            (110, y_pos + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        # Add HSV values
        hsv_text = f"HSV: ({avg_h}, {avg_s}, {avg_v})"
        cv2.putText(
            legend_img,
            hsv_text,
            (110, y_pos + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1
        )
        
        y_pos += row_height
    
    # Add a title and explanation
    title_img = np.ones((40, img_width, 3), dtype=np.uint8) * 255
    cv2.putText(
        title_img,
        "Legend - Color Clusters",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    cv2.putText(
        title_img,
        "Actual Color | Box Color",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
    
    # Combine title and legend
    full_legend = np.vstack([title_img, legend_img])
    
    return full_legend

# Create the enhanced legend
legend_img = create_enhanced_legend(groups, cluster_colors, img_width=300)

# Save both the annotated image and the legend
name = "backtoog"
cv2.imwrite(f"{name}.jpg", img)
cv2.imwrite(f"{name}_legend.jpg", legend_img)