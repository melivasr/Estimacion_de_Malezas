import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import pipeline

INPUT_DIR  = r"Pastos-2/pastos_test_images"   
OUTPUT_DIR = r"runs/depth_viz"                
MODEL_ID   = "Intel/dpt-large"                
DEVICE     = "auto"                           
PERC_CLIP  = (2, 98)                          


EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

def list_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in EXTS])

def ensure_dir(p: Path): 
    p.mkdir(parents=True, exist_ok=True)

def normalize_percentile(d: np.ndarray, lo_p=2, hi_p=98) -> np.ndarray:
    lo = np.percentile(d, lo_p); hi = np.percentile(d, hi_p)
    if hi - lo < 1e-9: 
        return np.zeros_like(d, np.float32)
    return ((np.clip(d, lo, hi) - lo) / (hi - lo)).astype(np.float32)

def apply_colormap(depth01: np.ndarray) -> np.ndarray:
    d8 = np.clip(depth01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)

# Profundidad (Transformers) 
def build_depth_pipeline(model_id: str, device_str: str = "auto"):
    if device_str == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif device_str == "cuda":
        device = 0
    else:
        device = -1
    # Task oficial: "depth-estimation"
    return pipeline(task="depth-estimation", model=model_id, device=device)

def infer_depth01(pipe, bgr: np.ndarray, clip_perc: Tuple[int,int]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = pipe(pil)                           # {"depth": PIL.Image, ...}
    depth_vis = np.array(out["depth"]).astype(np.float32)   # típico 0..255 visual
    depth01   = normalize_percentile(depth_vis, lo_p=clip_perc[0], hi_p=clip_perc[1])
    return depth01

if __name__ == "__main__":
    in_dir  = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    ensure_dir(out_dir)

    print(f"[INFO] Cargando: {MODEL_ID}")
    pipe = build_depth_pipeline(MODEL_ID, DEVICE)

    imgs = list_images(in_dir)
    if not imgs:
        raise FileNotFoundError(f"No hay imágenes en: {in_dir}")

    for p in imgs:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print("No se puede leer la imagen:", p); 
            continue

        depth01 = infer_depth01(pipe, bgr, PERC_CLIP)

        base = p.stem
        viz  = apply_colormap(depth01)
        cv2.imwrite(str(out_dir / f"{base}_viz.png"), viz)

        print(f" {base}")

    print("\n[LISTO] Carpeta:", out_dir.resolve())
