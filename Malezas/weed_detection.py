from roboflow import Roboflow
from ultralytics import YOLO
import cv2

rf = Roboflow(api_key="2PPZvnfTldc94GRa1W1Z")
project = rf.workspace("tec-h70pj").project("malezas-detection-uejg1")
version = project.version(1)
dataset = version.download("yolov11")
                
model = YOLO("yolo11n.pt") 
model.train(
    data=dataset.location + "/data.yaml",  # Roboflow exported dataset
    epochs=50,
    imgsz=640,
    batch=8,   
    workers=4
)

#Train model
trained_model = YOLO("runs/detect/train/weights/best.pt")

results = trained_model.predict("test_image.jpg", save=True, conf=0.5)

# show result
for r in results:
    img = r.plot() 
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
