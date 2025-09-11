from ultralytics import YOLO
import cv2, json 
import colordetector as cd 
from sklearn.cluster import AgglomerativeClustering
import preprocess 
import numpy as np
model = YOLO("../../runs/detect/train9/weights/best.pt")
img_path = "./corec/4.png"
raw = cv2.imread(img_path)
img = raw

results = model.predict(img, conf=0.5, iou=0.3, save=True, verbose=False)

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
def groups_by_hsv(detections, threshold=15): 
    groups = []
    for d in detections: 
        h,s,v = d["hsv"]
        matched = False
        for g in groups: 
            gh, gs, gv = g["hsv"]
            if abs(gh - h) < threshold and abs(gs - s) < threshold and abs(gv - v) < threshold: 
                g["members"].append(d)
                matched = True
                break
        if not matched: 
            groups.append({"hsv": (h,s,v), "members":[d]})
    return groups

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
    normalized_hsv[:, 0] = hsv_values[:, 0] / 180.0  # H component
    normalized_hsv[:, 1] = hsv_values[:, 1] / 255.0  # S component
    normalized_hsv[:, 2] = hsv_values[:, 2] / 255.0  # V component
    
    # Weight hue more heavily than saturation and value
    normalized_hsv[:, 0] *= 2.0
    
    print(json.dumps(detections[:3], indent=2))
    
    # Agglomerative clustering with distance_threshold
    clustering = AgglomerativeClustering(
        distance_threshold=0.3,  # Higher threshold for normalized HSV space
        n_clusters=None,
        linkage='ward',  # Ward linkage tends to create more balanced clusters
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

# Create a small legend showing each cluster's representative color and sample
legend_width = 250
legend_img = np.ones((len(groups) * 30 + 20, legend_width, 3), dtype=np.uint8) * 255

# Sort clusters by size (largest first) for the legend
sorted_clusters = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

for i, (index, group) in enumerate(sorted_clusters):
    color = cluster_colors[index]
    
    # Draw cluster color
    cv2.rectangle(legend_img, (10, i*30+10), (40, i*30+30), color, -1)
    
    # Draw text with cluster number and size
    cv2.putText(legend_img, f'Cluster {index}: {len(group)} holds', 
                (50, i*30+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    # Show most common color name in group
    if len(group) > 0:
        color_names = [d.get("color_name", "unknown") for d in group]
        most_common = max(set(color_names), key=color_names.count) if color_names else "unknown"
        if most_common != "unknown":
            cv2.putText(legend_img, f'({most_common})', 
                        (170, i*30+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)

# Save both the annotated image and the legend
cv2.imwrite("backtohsv.jpg", img)
cv2.imwrite("backtohsv_legend.jpg", legend_img)