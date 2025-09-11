# Average color clustering comparison
from ultralytics import YOLO
import cv2, json 
import colordetector as cd 
from sklearn.cluster import AgglomerativeClustering
import preprocess 
import numpy as np
import os

def get_average_color(img_crop):
    """Simple function to get average color of an image crop"""
    # Convert to HSV which is better for color comparison
    hsv_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Create a mask to exclude very dark or very bright areas
    h, s, v = cv2.split(hsv_img)
    mask = (s > 20) & (v > 30) & (v < 220)
    
    # If we have a valid mask with enough pixels
    if mask.sum() > 10:
        # Get average HSV values of the masked region
        avg_h = np.mean(h[mask]) if mask.any() else 0
        avg_s = np.mean(s[mask]) if mask.any() else 0
        avg_v = np.mean(v[mask]) if mask.any() else 0
    else:
        # Fallback to full image average if mask is too small
        avg_h = np.mean(h)
        avg_s = np.mean(s)
        avg_v = np.mean(v)
        
    return (avg_h, avg_s, avg_v)

def main():
    model = YOLO("../../runs/detect/train9/weights/best.pt")
    img_path = "./corec/4.png"
    raw = cv2.imread(img_path)
    img = raw.copy()

    results = model.predict(img, conf=0.5, iou=0.3, save=False, verbose=False)
    r = results[0]                       
    names = model.model.names          

    # Extract detections
    detections = []
    for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        
        # Skip invalid crops
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
            
        # Get average color
        avg_hsv = get_average_color(crop)
        
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "conf": float(conf),
            "class_id": int(cls),
            "class_name": names[int(cls)],
            "avg_hsv": avg_hsv
        })

    # Cluster using average colors
    if len(detections) > 0:
        avg_hsv_values = np.array([d["avg_hsv"] for d in detections])
        
        # Normalize HSV values for clustering
        normalized = np.zeros_like(avg_hsv_values, dtype=float)
        normalized[:, 0] = avg_hsv_values[:, 0] / 180.0  # H component
        normalized[:, 1] = avg_hsv_values[:, 1] / 255.0  # S component
        normalized[:, 2] = avg_hsv_values[:, 2] / 255.0  # V component
        
        # Weight hue more heavily than saturation and value
        normalized[:, 0] *= 2.0
        
        # Agglomerative clustering with distance_threshold
        clustering = AgglomerativeClustering(
            distance_threshold=0.3,
            n_clusters=None,
            linkage='ward',
            metric='euclidean'
        )
        labels = clustering.fit_predict(normalized)
        
        # Store cluster info
        for d, label in zip(detections, labels):
            d["cluster"] = int(label)
        
        # Group by cluster
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
        
        # Draw output image
        for index, group in groups.items():
            color = cluster_colors[index]
            for d in group:
                x1, y1, x2, y2 = map(int, d["bbox"])
                label = f'{index}'
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Create legend
        legend_width = 250
        legend_img = np.ones((len(groups) * 30 + 20, legend_width, 3), dtype=np.uint8) * 255
        
        # Sort clusters by size
        sorted_clusters = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (index, group) in enumerate(sorted_clusters):
            color = cluster_colors[index]
            
            # Draw cluster color
            cv2.rectangle(legend_img, (10, i*30+10), (40, i*30+30), color, -1)
            
            # Draw text with cluster number and size
            cv2.putText(legend_img, f'Cluster {index}: {len(group)} holds', 
                        (50, i*30+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
            # Get average HSV value for the cluster
            avg_h = np.mean([d["avg_hsv"][0] for d in group])
            avg_s = np.mean([d["avg_hsv"][1] for d in group])
            avg_v = np.mean([d["avg_hsv"][2] for d in group])
            
            # Show HSV values
            hsv_text = f'H:{avg_h:.1f} S:{avg_s:.1f} V:{avg_v:.1f}'
            cv2.putText(legend_img, hsv_text, 
                        (170, i*30+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)
        
        # Save output files
        cv2.imwrite("average_color_clustering.jpg", img)
        cv2.imwrite("average_color_legend.jpg", legend_img)
        
        print(f"Clustering complete. Found {len(groups)} clusters.")
        print(f"Images saved as 'average_color_clustering.jpg' and 'average_color_legend.jpg'")

if __name__ == "__main__":
    main()
