from ultralytics import YOLO
import cv2, json 

model = YOLO("../../runs/detect/train9/weights/best.pt")
img_path = "./corec/5.png"

results = model.predict(source=img_path, conf=0.6, iou=0.01, save=True, verbose=False)

r = results[0]                       
names = model.model.names          


boxes = r.boxes.xyxy.cpu().numpy()   
confs = r.boxes.conf.cpu().numpy()
clses = r.boxes.cls.cpu().numpy() 


detections = []
for (x1,y1,x2,y2), conf, cls in zip(boxes, confs, clses):
    detections.append({
        "bbox":[float(x1), float(y1), float(x2), float(y2)],
        "conf": float(conf),
        "class_id": int(cls),
        "class_name": names[int(cls)]
    })

print(json.dumps(detections[:3], indent=2))
img = cv2.imread(img_path)
for d in detections:
    x1,y1,x2,y2 = map(int, d["bbox"])
    label = f'{d["class_name"]} {d["conf"]:.2f}'
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
cv2.imwrite("wall_annotated.jpg", img)