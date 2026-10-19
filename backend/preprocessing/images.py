"""In-memory decoding and the SAME affine crop for training and inference. No face detector."""
import base64
import io
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
FORMATS = {"JPEG", "PNG", "WEBP"}
INPUT_SIZE = 96
CROP_MARGIN = 0.12


def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Размер изображения должен быть от 1 байта до 10 МБ.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in FORMATS:
                    raise ValueError("Поддерживаются только JPEG, PNG и WebP.")
                if image.width * image.height > MAX_PIXELS or min(image.size) < 32:
                    raise ValueError("Допустимый размер — от 32 × 32 пикселей до 20 мегапикселей.")
                if getattr(image,"is_animated",False):
                    raise ValueError("Анимированные изображения не поддерживаются.")
                return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Не удалось безопасно декодировать фото. Используйте корректный JPEG, PNG или WebP.") from exc


def encode_image(rgb: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def crop_image(rgb: np.ndarray, box: list[float], size: int = INPUT_SIZE, *, allow_outside: bool = False) -> tuple[np.ndarray, dict]:
    x,y,w,h = map(float,box)
    ih,iw = rgb.shape[:2]
    if not np.isfinite([x,y,w,h]).all() or min(w,h) < (1 if allow_outside else 24):
        raise ValueError("Выделите область лица не меньше 24 × 24 пикселей.")
    if not allow_outside and (x < 0 or y < 0 or x+w > iw+0.01 or y+h > ih+0.01):
        raise ValueError("Область лица должна находиться внутри изображения.")
    side = max(w,h) * (1+2*CROP_MARGIN)
    origin = [x+w/2-side/2, y+h/2-side/2]
    scale = (size-1)/side
    matrix = np.array([[scale,0,-origin[0]*scale],[0,scale,-origin[1]*scale]],np.float32)
    crop = cv2.warpAffine(rgb,matrix,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return crop, {"origin":origin,"side":side,"size":size,"margin":CROP_MARGIN,
                  "affine":matrix.tolist(),"normalization":"RGB / 127.5 − 1","border":"reflect101"}


def normalized_points(points: np.ndarray, transform: dict) -> np.ndarray:
    return (points - np.asarray(transform["origin"])) / transform["side"]


def image_points(points: np.ndarray, transform: dict) -> np.ndarray:
    return points * transform["side"] + np.asarray(transform["origin"])


def image_quality(crop: np.ndarray) -> dict:
    gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray,cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    messages = []
    if sharpness < 40:
        messages.append("Низкая резкость уменьшенной области. Размытие может смещать ориентиры.")
    if brightness < 45 or brightness > 215:
        messages.append("Слишком тёмное или яркое освещение может скрывать границы лица.")
    if contrast < 20:
        messages.append("Низкий контраст может сделать локализацию ориентиров нестабильной.")
    return {"laplacian_variance":round(sharpness,3),"mean_luminance":round(brightness,3),
            "luminance_std":round(contrast,3),"warnings":messages,
            "interpretation":"Это только признаки качества изображения, а не откалиброванная вероятность точности ориентиров."}
