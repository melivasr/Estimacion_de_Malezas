from ultralytics import YOLO

# Modelo entrenado train2
m = YOLO(r"runs/train_cls_pastos5/weights/best.pt")

# Lista de imágenes
images = r"Pastos-2/pastos_test_images"

# Ejecuta predicciones
results = m.predict(source=images, conf=0.5, save=True, show=False)
