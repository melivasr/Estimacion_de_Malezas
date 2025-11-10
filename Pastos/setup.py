from pathlib import Path
from roboflow import Roboflow
import os

API_KEY   = "2PPZvnfTldc94GRa1W1Z"
WORKSPACE = "tec-h70pj"
PROJECT   = "pastos-qtg7k"
VERSION   = 2
FORMAT    = "folder"  

DATA_DIR_FILE = "DATA_DIR.txt"

if __name__ == "__main__":
    print("loading Roboflow workspace...")
    rf = Roboflow(api_key=API_KEY)

    print("loading Roboflow project...")
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    dataset = version.download(FORMAT)     
    dataset_dir = Path(dataset.location)  
    print(" Dataset descargado en:", dataset_dir)

   
    Path(DATA_DIR_FILE).write_text(str(dataset_dir), encoding="utf-8")
    print(f" Ruta guardada en: {DATA_DIR_FILE}")
