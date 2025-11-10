
"""
Este código entrena un modelo YOLO para segmentación de malezas usando un dataset descargado desde Roboflow.

1. Carga el modelo base especificado en BASE_MODEL.
2. Entrena el modelo usando el dataset definido en DATA_YAML.
3. Configura los parámetros de entrenamiento: epochs, tamaño de imagen, batch, device, etc.
4. Guarda los pesos entrenados en la carpeta 'runs/train_seg/weights/'.
5. Imprime la ubicación de los pesos finales.

"""
import os
from ultralytics import YOLO

DATA_YAML = r"Malezas-detection-2-1/data.yaml"  
BASE_MODEL = "yolo11n-seg.pt"                  

EPOCHS = 15
IMG_SIZE = 640
BATCH = 10
WORKERS = 0
DEVICE = "cpu"       

if __name__ == "__main__":
    print("Cargando modelo base:", BASE_MODEL)
    model = YOLO(BASE_MODEL)

    print("Entrenando con:", DATA_YAML)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project="runs",
        name="train_seg",
        patience=20,
        pretrained=True
    )

    print("\n Pesos en: runs/train_seg/weights/best.pt")
