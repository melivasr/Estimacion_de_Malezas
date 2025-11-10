"""
Código de entrenamiento de modelo YOLO para clasificación de pastos.

Realiza los siguientes pasos:
1. Lee la ruta del dataset desde el archivo DATA_DIR.txt.
2. Carga un modelo base YOLO para clasificación.
3. Entrena el modelo usando los parámetros definidos.
4. Los resultados del entrenamiento se guardan en la carpeta `runs`.

- Para obtener resultados actualizados, se debe ejecutar nuevamente este script.
- Asegúrese de haber descargado previamente el dataset y creado DATA_DIR.txt.
"""
from ultralytics import YOLO
from pathlib import Path
import os

BASE_MODEL = "yolo11n-cls.pt"   
IMGSZ      = 224
EPOCHS     = 50
BATCH      = 16     
WORKERS    = 0       
DEVICE     = "cpu"   

PROJECT = "runs"
RUN_NAME = "train_cls_pastos"

DATA_DIR_FILE = "DATA_DIR.txt"

def get_dataset_dir():
    if Path(DATA_DIR_FILE).is_file():
        p = Path(DATA_DIR_FILE).read_text(encoding="utf-8").strip()
        return Path(p)
    raise FileNotFoundError(
        f"No hay {DATA_DIR_FILE}. Ejecuta primero 01_get_data_cls.py para descargar el dataset."
    )

if __name__ == "__main__":
    data_dir = get_dataset_dir()
    print(" Dataset dir:", data_dir)

    
    print(" Cargando modelo base:", BASE_MODEL)
    model = YOLO(BASE_MODEL)

    print(" Entrenando (clasificación)...")
    model.train(
        data=str(data_dir),    
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,
        val=True,
        erasing=0.0,            
        auto_augment=None,      
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,  
        translate=0.0, scale=0.0, degrees=0.0, 
        fliplr=0.0          
    )

    print("\nRevisa resultados en:", Path(PROJECT) / RUN_NAME)
