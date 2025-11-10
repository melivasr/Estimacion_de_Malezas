from ultralytics import YOLO

# Modelo entrenado train2
m = YOLO(r"runs/detect/train2/weights/best.pt")

# Lista de imágenes
images = r"Malezas-detection-1/test_images_train"

# Ejecuta predicciones
results = m.predict(source=images, conf=0.5, save=True, show=False)


