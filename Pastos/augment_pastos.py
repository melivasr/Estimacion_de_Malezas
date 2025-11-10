"""
Código para aumentar la cantidad de imágenes de pastos usando Albumentations.

1. Busca todas las subcarpetas de clases dentro de ROOT_DIR.
2. Para cada imagen, aplica transformaciones de aumento:
   - RandomResizedCrop
   - HorizontalFlip
   - ShiftScaleRotate
   - RandomBrightnessContrast
3. Guarda las nuevas imágenes en la misma carpeta con sufijo `_augN`.
4. Devuelve el total de imágenes creadas.

"""

from pathlib import Path
from typing import List, Set
import cv2
from tqdm import tqdm
import albumentations as A
import numpy as np

# Configuración
ROOT_DIR = Path(r"Pastos-2/train")
N_AUG_PER_IMAGE = 3
ONLY_CLASSES: List[str] = []  # lista de nombres de subcarpetas; vacío = todas
SEED = 42

EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

AUG_PIPE = A.Compose(
    [
        A.RandomResizedCrop(size=(480, 480), scale=(0.75, 1.0), p=0.6),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15, rotate_limit=20, p=0.6, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(p=0.5),
    ],
    p=1.0,
)


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS


def get_class_dirs(root: Path, only: List[str]) -> List[Path]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Ruta no encontrada: {root}")
    class_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if only:
        class_dirs = [d for d in class_dirs if d.name in set(only)]
    return class_dirs


def next_aug_index(folder: Path, stem: str, suffix: str) -> int:
    n = 0
    while True:
        candidate = folder / f"{stem}_aug{n}{suffix}"
        if not candidate.exists():
            return n
        n += 1


def augment_pastos(root_dir: Path, n_aug_per_image: int, only_classes: List[str] = None, seed: int = 42) -> int:
    """Aumenta imágenes en cada subcarpeta de `root_dir`. Devuelve total creadas."""
    if only_classes is None:
        only_classes = []
    np.random.seed(seed)

    class_dirs = get_class_dirs(root_dir, only_classes)
    if not class_dirs:
        print("No se encontraron subcarpetas de clase en:", root_dir)
        return 0

    total_created = 0
    for cdir in class_dirs:
        imgs = [p for p in sorted(cdir.iterdir()) if is_image_file(p)]
        if not imgs:
            continue
        for img_path in tqdm(imgs, desc=f"Aumentando {cdir.name}", unit="img"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f" No se pudo leer: {img_path}")
                continue
            for _ in range(n_aug_per_image):
                try:
                    aug = AUG_PIPE(image=img)
                    out_img = aug["image"]
                except Exception as e:
                    print(f"Aug fallo para {img_path.name}: {e}. Usando copia original.")
                    out_img = img.copy()
                idx = next_aug_index(cdir, img_path.stem, img_path.suffix)
                out_name = f"{img_path.stem}_aug{idx}{img_path.suffix}"
                out_path = cdir / out_name
                if cv2.imwrite(str(out_path), out_img):
                    total_created += 1
                else:
                    print(f" No se pudo escribir: {out_path}")
    return total_created


if __name__ == "__main__":
    created = augment_pastos(ROOT_DIR, N_AUG_PER_IMAGE, ONLY_CLASSES, SEED)
    print(f"Total imágenes aumentadas: {created}")
    print("Nuevas imágenes añadidas en las subcarpetas bajo:", ROOT_DIR.resolve())