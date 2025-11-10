"""
Este código descarga el dataset de segmentación de malezas desde Roboflow.

1. Conecta con Roboflow usando la API_KEY.
2. Selecciona el workspace y proyecto específicos.
3. Obtiene la versión indicada del proyecto.
4. Descarga el dataset en formato YOLOv11.
5. Imprime la ruta local donde se descargó el dataset.
"""
from roboflow import Roboflow

API_KEY = "2PPZvnfTldc94GRa1W1Z"
WORKSPACE = "tec-h70pj"
PROJECT = "malezas-detection-2-tuj7l"   

if __name__ == "__main__":
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(1)
    dataset = version.download("yolov11")
    print(" Dataset descargado en:", dataset.location)

