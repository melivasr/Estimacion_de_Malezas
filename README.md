
# Proyecto de Detección y Clasificación de Malezas y Pastos

Este proyecto consiste en entrenar modelos de detección y clasificación para malezas y pastos. Debido a la gran cantidad de imágenes y archivos generados, subir todo el proyecto a GitHub es un proceso pesado, por lo que a continuación puede encontrar una guía de como ejecutar los códigos.

## Estructura del Proyecto

```bash
.
├── Malezas/
│   ├── weed_detection.py
│   └── test.py
├── Malezas_Segmentation/
│   ├── predict_dept.py
│   ├── predict_seg.py
│   ├── setup.py
│   └── train_seg.py
├── Pastos/
│   ├── augment_pastos.py
│   ├── predict_dept.py
│   ├── setup.py
│   ├── test.py
│   └── train.py
└── README.md
```


## Requisitos del sistema

- Sistema operativo: **Windows**
- IDE recomendado: **VS Code**
- Python instalado (recomendado >=3.9)
- 
Antes de ejecutar los scripts, asegúrate de tener instaladas las siguientes librerías:

```bash
pip install ultralytics roboflow opencv-python albumentations tqdm torch torchvision transformers pillow
```

---

## Malezas

### 1. Entrenamiento del modelo de detección
**Archivo:** `Malezas/weed_detection.py`

Este script hace lo siguiente:
- Descarga el dataset desde Roboflow.
- Carga el modelo base YOLO.
- Entrena el modelo durante 50 epochs.
- Realiza predicciones sobre imágenes de prueba.

**Ejecutar:**
```bash
python Malezas/weed_detection.py
```

### 2. Predicciones con modelo entrenado
**Archivo:** `Malezas/test.py`

- Carga el modelo entrenado (`runs/detect/train2/weights/best.pt`).
- Realiza predicciones sobre imágenes de prueba en `Malezas-detection-1/test_images_train`.

**Ejecutar:**
```bash
python Malezas/test.py
```

---

## Malezas_Segmentation

### 1. Descargar dataset desde Roboflow

**Archivo:** `Malezas_Segmentation/setup.py`

* Descarga el dataset de segmentación desde Roboflow.

**Ejecutar:**

```bash
python Malezas_Segmentation/setup.py
```

---

### 2. Entrenamiento del modelo de segmentación

**Archivo:** `Malezas_Segmentation/train_seg.py`

* Usa el dataset descargado.
* Entrena un modelo YOLO para segmentación de malezas.
* Configurable base model, epochs, tamaño de imagen, batch, device, etc.

**Ejecutar:**

```bash
python Malezas_Segmentation/train_seg.py
```

---

### 3. Predicciones de segmentación

**Archivo:** `Malezas_Segmentation/predict_seg.py`

* Carga el modelo entrenado (`runs/train_seg/weights/best.pt`).
* Realiza predicciones sobre imágenes de prueba en `Malezas-detection-2-1/test_images_train`.
* Los resultados se guardan en `runs/segment/predict*/`.

**Ejecutar:**

```bash
python Malezas_Segmentation/predict_seg.py
```

---

### 4. Estimación de profundidad para segmentación

**Archivo:** `Malezas_Segmentation/predict_dept.py`

* Estima la profundidad de las imágenes usando `Intel/dpt-large`.
* Genera mapas de profundidad visualizados en `runs/depth_viz`.

**Ejecutar:**

```bash
python Malezas_Segmentation/predict_dept.py
```

---


## Pastos


### 1. Descargar dataset desde Roboflow
**Archivo:** `Pastos/setup.py`

- Descarga el dataset desde Roboflow.
- Guarda la ruta en `DATA_DIR.txt` para usar en el entrenamiento.

**Ejecutar:**
```bash
python Pastos/setup.py
```

### 2. Aumento de imágenes
**Archivo:** `Pastos/augment_pastos.py`

- Realiza aumentos de datos (rotación, flip, recorte, brillo/contraste) sobre todas las subcarpetas de `Pastos-2/train`.
- Configurable el número de aumentos por imagen (`N_AUG_PER_IMAGE`).

**Ejecutar:**
```bash
python Pastos/augment_pastos.py
```

### 3. Entrenamiento del modelo de clasificación
**Archivo:** `Pastos/train.py`

- Usa el dataset descargado y la ruta de `DATA_DIR.txt`.
- Entrena un modelo YOLO para clasificación de pastos.
- Configurable base model, tamaño de imagen, epochs, batch, device, etc.

**Ejecutar:**
```bash
python Pastos/train.py
```

### 4. Predicciones con modelo entrenado
**Archivo:** `Pastos/test.py`

- Carga el modelo entrenado (`runs/train_cls_pastos/weights/best.pt`), esta ruta puede ajustarse dependiendo del entrenamiento.
- Realiza predicciones sobre imágenes de prueba en `Pastos-2/pastos_test_images`.

**Ejecutar:**
```bash
python Pastos/test.py
```

### 5. Estimación de profundidad
**Archivo:** `Pastos/predict_dept.py`

- Estima la profundidad de las imágenes usando `Intel/dpt-large` con la librería `transformers`.
- Genera mapas de profundidad visualizados en `runs/depth_viz`.

**Ejecutar:**
```bash
python Pastos/predict_dept.py
```

---

## Orden recomendado de ejecución

**Malezas:**
1. `weed_detection.py`
2. `test.py`
   
**Malezas_Segmentation:**

1. `setup.py`
2. `train_seg.py`
3. `predict_seg.py`
4. `predict_dept.py`

**Pastos:**
1. `augment_pastos.py`
2. `setup.py`
3. `train.py`
4. `test.py`
5. `predict_dept.py`

---

## Notas

- Asegúrese de tener suficientes recursos de hardware, ya que algunos scripts (entrenamiento y aumento de datos) pueden ser intensivos en CPU/GPU y memoria.
