"""
Este código realiza predicciones de segmentación de malezas usando un modelo YOLO entrenado.

Pasos principales:
1. Carga los pesos del modelo entrenado (WEIGHTS).
2. Lista todas las imágenes en IMAGES_DIR.
3. Realiza predicciones de segmentación sobre todas las imágenes:
   - Genera imágenes con máscaras superpuestas.
   - Guarda resultados en 'runs/segment/predict*/'.
   - Guarda archivos TXT si corresponde.
4. Indica la ubicación de los resultados.
"""
import os
from glob import glob
from ultralytics import YOLO

WEIGHTS = r"runs/train_seg3/weights/best.pt"         
IMAGES_DIR = r"Malezas-detection-2-1/test_images_train" 
CONF = 0.5
IMG_SIZE = 640
DEVICE = "cpu"  

def main():
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"No hay pesos: {WEIGHTS}")

    model = YOLO(WEIGHTS)
    images = sorted(glob(os.path.join(IMAGES_DIR, "*.*")))
    if not images:
        raise FileNotFoundError(f"No hay imágenes en: {IMAGES_DIR}")

    print(f" {len(images)} imágenes encontradas. Prediciendo...")
    model.predict(
        source=IMAGES_DIR,
        conf=CONF,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=True,   
        save_txt=True,   
        save_conf=True,
        retina_masks=True
    )
    print("Resultados se encuentran en la carpeta runs/segment/predict*/")

if __name__ == "__main__":
    main()