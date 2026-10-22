"""Local research pipeline: Face Landmarker + SCUT human-rating regressor.

The scalar prediction and descriptive geometry are independent outputs.
No region scores, percentiles, or causal explanations are fabricated.
"""
import hashlib
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.geometry.metrics import measure
from backend.ml.inference import ModelUnavailable
from backend.ml.personal import HEAD_MARGINS, PersonalHead
from backend.preprocessing.images import crop_image, decode_image, encode_image, image_quality
from scripts.download_holistic_models import MODELS

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "training/checkpoints/holistic"
MODEL_ID = "evanlyhf/scut-fbp5500-beauty"
REVISION = "44c77683f3640a259aacff59a2d635875ec65729"


def display_score(raw: float) -> float:
    return round(float(np.clip((raw - 1) * 2.5, 0, 10)), 2)


def aligned_crop(rgb: np.ndarray, points: np.ndarray, margin: float) -> np.ndarray:
    """Square contour crop, rotated by the pupil line; reflected border padding."""
    left, right = points[468], points[473]
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    center = (left + right) / 2
    rotation = cv2.getRotationMatrix2D(tuple(center), angle, 1)
    contour = points[[10,127,234,93,172,152,397,323,454,356]]
    rotated = contour @ rotation[:, :2].T + rotation[:, 2]
    low, high = rotated.min(0), rotated.max(0)
    side = float(max(high - low) * (1 + margin * 2))
    origin = (low + high) / 2 - side / 2
    scale = 223 / side
    rotation[:, 2] -= origin
    rotation *= scale
    return cv2.warpAffine(rgb, rotation, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def regional_report(points: np.ndarray, geometry: dict, rgb: np.ndarray) -> list[dict]:
    metrics = {row["id"]: row for row in geometry["metrics"]}

    def item(key):
        row = metrics[key]
        return {"id": key, "title": row["title"], "value": row["value"], "unit": row["unit"], "point_ids": row["point_ids"]}

    def value(key):
        v = metrics[key]["value"]
        return f"{v:.2f}" if v is not None else "недоступно"

    groups = [
        ("eyes", "Глаза", "Учитываем оба глаза отдельно", ["eye_tilt_0", "eye_tilt_1", "eye_aspect_0", "eye_aspect_1", "eye_tilt_difference", "eye_spacing"],
         f"Наклон уголков: {value('eye_tilt_0')}° и {value('eye_tilt_1')}°. Разница — {value('eye_tilt_difference')}°. Раскрытие век зависит от мимики и направления взгляда."),
        ("brows", "Брови и область глаз", "Положение и видимый изгиб", ["brow_gap_0", "brow_gap_1", "brow_angle_0", "brow_angle_1"],
         "Измерены расстояния от бровей до глаз и направления бровей. Густота, цвет и глубина посадки глаз этими точками не измеряются."),
        ("upper", "Лоб и верхняя область", "Видимый контур, без линии роста волос", ["visible_upper_middle"],
         f"Видимая верхняя область к среднему сегменту — {value('visible_upper_middle')}. Линия роста волос не размечена; классические трети лица по этому показателю не определяются."),
        ("midface", "Средняя область лица", "Расположение глаз, носа и губ", ["midface_pupils", "width_height", "pupil_face"],
         f"Отношение межзрачкового расстояния к высоте до верхней губы — {value('midface_pupils')}. Ширина контура к высоте от переносицы — {value('width_height')}."),
        ("nose_mouth", "Нос и рот", "Сочетание видимых ширин", ["mouth_nose", "mouth_pupils"],
         f"Ширина рта / ширина носа — {value('mouth_nose')}. Измерение описывает фронтальный снимок; проекция носа и губ в профиль здесь недоступна."),
        ("lower", "Челюсть и подбородок", "Контур нижней области", ["jaw_width", "lower_middle", "chin_lip"],
         f"Отношение нижней ширины контура к ширине лица — {value('jaw_width')}. Нижний сегмент к среднему — {value('lower_middle')}. Это видимый контур, без оценки костей и профильных углов."),
        ("symmetry", "Сочетание и симметрия", "Парные ориентиры в плоскости фото", ["asymmetry", "eye_aspect_difference"],
         f"Среднее расхождение выбранных парных ориентиров после отражения — {float(metrics['asymmetry']['value']) * 100:.1f}% ширины лица. Поворот головы, мимика и ошибка точек тоже создают различия."),
    ]
    regions = [{"id": key, "title": title, "subtitle": subtitle, "summary": summary, "metrics": [item(k) for k in keys], "kind": "measured_geometry"}
               for key, title, subtitle, keys, summary in groups]

    # Photometric observations only. These statistics cannot establish skin
    # health or a learned skin-attractiveness rating.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    radius = max(3, int(np.linalg.norm(points[234] - points[454]) * .055))
    mask = np.zeros(gray.shape, np.uint8)
    for point in points[[205,425,151]]:
        cv2.circle(mask, tuple(np.rint(point).astype(int)), radius, 255, -1)
    smooth = cv2.GaussianBlur(gray, (0,0), max(1, radius / 3))
    samples = smooth[mask > 0]
    spread = round(float(np.percentile(samples, 75) - np.percentile(samples, 25)) / 255 * 100, 2)
    regions.append({"id": "skin", "title": "Кожа и освещение", "subtitle": "Наблюдения по изображению", "kind": "photometric_observation",
                    "summary": f"Разброс яркости небольших участков щёк и лба — {spread:.1f}% шкалы яркости. На него влияют тени, макияж, обработка и фактура. Оценка здоровья и отдельный балл кожи не рассчитываются.",
                    "metrics": [{"id": "skin_luminance_iqr", "title": "Межквартильный размах яркости участков", "value": spread, "unit": "%", "point_ids": [205,425,151]}]})
    return regions


class HolisticAnalyzer:
    def __init__(self, model_dir: Path = MODEL_DIR, *, personal: bool | None = None):
        self.model_dir = model_dir
        self.lock = threading.RLock()
        self.model = None
        self.landmarker = None
        self.problem = None
        self.load_ms = None
        self.personal_requested = personal
        self.personal = None
        self.visual_block = None
        self.profile_loaded = False
        torch.set_num_threads(4)

    def _load(self):
        if self.model is not None and self.landmarker is not None and self.profile_loaded:
            return
        start = time.perf_counter()
        for name, (_, expected) in MODELS.items():
            path = self.model_dir / name
            if not path.is_file():
                raise ModelUnavailable("Не найдены веса комплексной оценки. Выполните setup.ps1 или python -m scripts.download_holistic_models.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ModelUnavailable(f"Контрольная сумма весов {name} не совпала. Повторите загрузку моделей.")
        try:
            import mediapipe as mp
            from torchvision.models import resnet50

            model = resnet50(weights=None, num_classes=1)
            state = torch.load(self.model_dir / "beauty_regressor.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError("Non-finite checkpoint parameters")
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_dir / "face_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
                min_face_detection_confidence=.6, min_face_presence_confidence=.6,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True,
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self.model = model.eval()
            metadata_path = self.model_dir / 'personal-preference.json'
            if self.personal_requested is not False and metadata_path.is_file():
                import json
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                if metadata.get('enabled') or self.personal_requested is True:
                    self.personal = PersonalHead(self.model_dir)
                    from backend.ml.personal_visual import load_visual_block
                    self.visual_block = load_visual_block(self.model_dir, metadata, model.layer4[2])
            elif self.personal_requested is True:
                raise ModelUnavailable('Персональные веса ещё не установлены.')
            self.profile_loaded = True
            self.load_ms = round((time.perf_counter() - start) * 1000, 2)
        except Exception as exc:
            raise ModelUnavailable(f"Не удалось загрузить модели комплексной оценки: {type(exc).__name__}: {exc}") from exc

    def status(self):
        with self.lock:
            try:
                self._load()
                self.problem = None
            except ModelUnavailable as exc:
                self.problem = str(exc)
            personal = self.personal.metadata if self.personal is not None else None
            return {"ready": self.model is not None and self.landmarker is not None and self.profile_loaded and self.problem is None, "reason": self.problem,
                    "model_id": personal['model_id'] if personal else MODEL_ID, "revision": personal['revision'] if personal else REVISION, "landmarks": 478, "device": "cpu",
                    "backbone_id": MODEL_ID,
                    "source_scale": [1,10] if personal else [1,5], "display_scale": [1,10] if personal else [0,10],
                    "display_formula": "Прямой прогноз личного балла" if personal else "(raw − 1) × 2.5",
                    "active_profile": "personal" if personal else "scut",
                    "personal": {**{key: personal[key] for key in ('model_id','revision','feature_mode','fit_images','test_images','total_rated_images','validation','identity_protocol')}, 'training_method':personal.get('training_method','frozen_head')} if personal else None,
                    "research_only": True, "external_ai": False, "photo_storage": False}

    def embed_crops(self, crops: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
        """Personal visual features; original SCUT predictions always stay intact."""
        batch = np.stack(crops).astype(np.float32) / 255
        batch = (batch - np.array([.485,.456,.406], np.float32)) / np.array([.229,.224,.225], np.float32)
        tensor = torch.from_numpy(batch.transpose(0,3,1,2).copy())
        with torch.inference_mode():
            model = self.model
            hidden = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
            for layer in (model.layer1, model.layer2, model.layer3, model.layer4[0], model.layer4[1]):
                hidden = layer(hidden)
            pooled = model.avgpool(model.layer4[2](hidden)).flatten(1)
            values = model.fc(pooled).flatten().tolist()
            if self.visual_block is not None:
                pooled = model.avgpool(self.visual_block(hidden)).flatten(1)
            features = pooled.numpy().copy()
        if not np.isfinite(values).all() or not np.isfinite(features).all():
            raise ModelUnavailable("Модель вернула некорректные значения; балл не рассчитан.")
        return features, values

    def predict_crops(self, crops: list[np.ndarray]) -> list[float]:
        return self.embed_crops(crops)[1]

    def analyze(self, data: bytes, box: list[float] | None = None):
        with self.lock:
            self._load()
            return self._analyze(data, box)

    def _analyze(self, data: bytes, box: list[float] | None):
        import mediapipe as mp

        start = time.perf_counter()
        rgb = decode_image(data)
        ih, iw = rgb.shape[:2]
        roi = rgb
        offset = np.zeros(2)
        if box is not None:
            crop_image(rgb, box)  # reuse finite/in-bounds validation
            x,y,w,h = box
            offset = np.array([int(x),int(y)])
            roi = rgb[int(y):math.ceil(y+h),int(x):math.ceil(x+w)]
        scale = min(1, 1600 / max(roi.shape[:2]))
        resized = cv2.resize(roi, (round(roi.shape[1]*scale), round(roi.shape[0]*scale))) if scale < 1 else roi
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(resized))
        detection = self.landmarker.detect(image)
        count = len(detection.face_landmarks)
        if count != 1:
            message = "Лицо не найдено. Выберите чёткий снимок анфас без перекрытий." if not count else "Найдено несколько лиц. Обведите одно лицо или загрузите отдельный портрет."
            return {"localization": {"status": "not_found" if not count else "multiple_faces", "message": message}, "analysis": None}
        points = np.array([[p.x*roi.shape[1],p.y*roi.shape[0]] for p in detection.face_landmarks[0]]) + offset
        contour = points[[10,127,234,93,172,152,397,323,454,356]]
        low, high = contour.min(0), contour.max(0)
        face_box = [float(low[0]),float(low[1]),float(high[0]-low[0]),float(high[1]-low[1])]
        if min(face_box[2:]) < 100 or np.linalg.norm(points[468]-points[473]) < 32:
            raise ValueError("Лицо слишком маленькое для комплексной оценки. Нужен более крупный и чёткий портрет.")
        if np.any(low < 1) or high[0] > iw-2 or high[1] > ih-2:
            raise ValueError("Контур лица обрезан краем снимка. Нужны лоб, оба глаза и подбородок целиком.")
        # The selected ROI must contain the inferred contour as well.
        if box is not None and (np.any(low < offset) or np.any(high > offset + np.array([roi.shape[1],roi.shape[0]]))):
            raise ValueError("Выделенная область обрезает лицо. Захватите лоб и подбородок целиком.")
        geometry = measure(points, "mediapipe")
        primary = aligned_crop(rgb, points, .20)
        quality = image_quality(primary)
        rotation = detection.facial_transformation_matrixes[0][:3,:3]
        pitch, yaw, _ = map(float, cv2.RQDecomp3x3(rotation)[0])
        roll = geometry["roll_degrees"]
        quality["pose"] = {"yaw": round(yaw,2), "pitch": round(pitch,2), "roll": round(roll,2), "kind": "model_pose_estimate"}
        if abs(yaw)>30 or abs(pitch)>30 or abs(roll)>30:
            raise ValueError("Сильный поворот или наклон головы. Для этой модели нужен снимок анфас с ровным положением головы.")
        if quality["laplacian_variance"]<10 or quality["mean_luminance"]<25 or quality["mean_luminance"]>240:
            raise ValueError("На фото слишком мало деталей: сильное размытие или неподходящий свет. Попробуйте другой снимок.")
        expressions = {v.category_name: v.score for v in detection.face_blendshapes[0]}
        eye_apertures = [row["value"] for row in geometry["metrics"] if row["id"] in {"eye_aspect_0","eye_aspect_1"}]
        if max(expressions.get("eyeBlinkLeft",0),expressions.get("eyeBlinkRight",0))>.75 or any(v is None or v>8 for v in eye_apertures):
            raise ValueError("Один или оба глаза закрыты либо сильно прищурены. Загрузите снимок с открытыми глазами.")
        warnings = list(quality["warnings"])
        if abs(yaw)>15 or abs(pitch)>15 or abs(roll)>15:
            warnings.append("Есть заметный наклон или поворот головы. Сравнивайте результат с более фронтальным снимком.")
        if max(expressions.get("mouthSmileLeft",0),expressions.get("mouthSmileRight",0),expressions.get("jawOpen",0))>.45:
            warnings.append("Выраженная мимика меняет форму глаз, рта и нижней области лица. Для сравнения нужен нейтральный портрет.")
        before_regression = time.perf_counter()
        crops = [aligned_crop(rgb,points,margin) for margin in (.12,.20,.28)]
        visual_crops = crops
        uses_head = self.personal is not None and self.personal.metadata['feature_mode'] in {'head','dual'}
        if uses_head:
            visual_crops = crops + [aligned_crop(rgb,points,margin) for margin in HEAD_MARGINS]
        features, predictions = self.embed_crops([c for crop in visual_crops for c in (crop,np.fliplr(crop).copy())])
        raw = float(np.mean(predictions[2:4]))
        if self.personal is None and (raw < 1 or raw > 5):
            raise ValueError("Прогноз модели вышел за исходную шкалу 1–5. Надёжно показать общий балл для этого снимка не удалось; попробуйте другой нейтральный портрет.")
        crop_means = [float(np.mean(predictions[i:i+2])) for i in (0,2,4)]
        spread = max(crop_means) - min(crop_means)
        flip_delta = abs(predictions[2]-predictions[3])
        personal_values = None
        personal_flip_delta = None
        if self.personal is not None:
            features = features.reshape(-1,2,2048)
            head_features = features[3:] if uses_head else features[:3]
            paired = np.stack((features[:3].mean(axis=1), head_features.mean(axis=1)), axis=1)
            personal_values = self.personal.predict(paired, (np.asarray(crop_means)-1)*2.5)
            flip_features = np.stack((features[1],head_features[1]),axis=1)
            flip_values = self.personal.predict(flip_features,(np.asarray(predictions[2:4])-1)*2.5)
            personal_flip_delta = float(abs(flip_values[0]-flip_values[1]))
        sensitive = max(spread,flip_delta)>.4 if personal_values is None else max(float(np.ptp(personal_values)),personal_flip_delta)>1
        if sensitive:
            warnings.append("Балл заметно меняется при небольшом изменении кадрирования или отражении. Результат этого снимка нестабилен.")
        quality["warnings"] = warnings
        quality["status"] = "review" if warnings else "suitable"
        regions = regional_report(points, geometry, rgb)
        crop, transform = crop_image(rgb, face_box, 224)
        regression_ms = round((time.perf_counter()-before_regression)*1000,2)
        elapsed = round((time.perf_counter()-start)*1000,2)
        holistic = {"status": "calculated", "value": display_score(raw), "raw_value": round(raw,4),
                    "scale": [0,10], "source_scale": [1,5], "conversion": "(raw − 1) × 2.5",
                    "kind": "predicted_mean_human_rating", "model_id": MODEL_ID, "revision": REVISION,
                    "interpretation": "Предсказание средней человеческой оценки по шкале SCUT-FBP5500. Это субъективная оценка снимка, а не объективная мера красоты.",
                    "validation": {"human_agreement_measured_here": False, "independent_identity_photos_available": False},
                    "stability": {"low": display_score(min(crop_means)), "high": display_score(max(crop_means)),
                                  "crop_raw_values": [round(v,4) for v in crop_means], "flip_raw_delta": round(flip_delta,4),
                                  "status": "sensitive" if sensitive else "stable_in_this_check",
                                  "interpretation": "Разброс при трёх отступах кадрирования 12/20/28% и отражении. Это техническая чувствительность, не доверительный интервал."},
                    "regions": regions,
                    "explanation": "Общий балл рассчитан визуальной нейросетью. Измерения областей описывают снимок отдельно и не доказывают причину этого балла.",
                    "limitations": ["Шкала отражает оценки участников исходного датасета; перенос на новые лица отдельно не проверен.",
                                    "Профиль, трёхмерная форма костей и линия роста волос по этому анализу не оцениваются.",
                                    "Фоторетушь, фильтры, свет и мимика могут менять прогноз."]}
        if self.personal is not None:
            metadata = self.personal.metadata
            holistic.update({
                'value': round(float(personal_values[1]),2), 'raw_value': round(float(personal_values[1]),4),
                'scale': [1,10], 'source_scale': [1,10], 'conversion': 'Прямой прогноз личной оценки 1–10',
                'kind': 'predicted_personal_rating', 'model_id': metadata['model_id'], 'revision': metadata['revision'],
                'interpretation': 'Прогноз ваших личных оценок фотографии. Основа — SCUT ResNet-50; ' + ('последний зрительный блок и регрессионная голова дообучены на вашей разметке.' if self.visual_block is not None else 'персональная регрессионная голова обучена на вашей разметке.'),
                'personal': {key:metadata[key] for key in ('feature_mode','fit_images','test_images','total_rated_images','validation')},
                'baseline': {'value':display_score(raw),'raw_value':round(raw,4),'model_id':MODEL_ID},
                'validation': {'human_agreement_measured_here':True,'independent_identity_photos_available':False,
                               'protocol':'Отложены целые группы визуально проверенных повторов. Полнота группировки людей не гарантирована; это небольшой тест одного оценщика.'},
                'stability': {'low':round(float(min(personal_values)),2),'high':round(float(max(personal_values)),2),
                              'crop_raw_values':np.round(personal_values,4).tolist(),'flip_raw_delta':round(personal_flip_delta,4),
                              'status':'sensitive' if sensitive else 'stable_in_this_check',
                              'interpretation':('Лицо и более широкий фрагмент головы' if uses_head else 'Фрагмент лица') + ', три отступа рамки и отражение. Технический разброс, не доверительный интервал.'},
                'explanation':('Последний зрительный блок ResNet-50 и персональная регрессионная голова дообучены по вашим оценкам. ' if self.visual_block is not None else 'Признаки обученной ResNet-50 переданы персональной регрессионной голове. Её веса подобраны по вашим оценкам. ') + 'Проверочные фото не участвовали в обучении. Геометрические измерения не складываются в этот балл.',
                'limitations':['Это приближение ваших предпочтений по небольшой выбранной подборке, а не общая шкала красоты.',
                               'Повторы людей проверены визуально; полное отсутствие пересечений с обучением базовой модели установить нельзя.',
                               'Фон, волосы, свет, мимика и ретушь могут влиять на прогноз. Проверка новых людей и снимков ещё ограничена.',
                               *([] if uses_head else ['В этой версии выбран фрагмент лица; часть причёски может не попадать в кадрирование.']),
                               'Профиль и трёхмерная форма костей не оцениваются.'],
            })
        localization = {"status": "found", "method": "mediapipe_face_landmarker_v1", "faces": 1}
        analysis = {"source": {"width": iw,"height": ih,"bytes": len(data)}, "box": face_box,
                    "preprocessed_image": encode_image(crop), "visual_model_image": encode_image(visual_crops[4] if uses_head else primary), "transform": transform,
                    "quality": quality, "localization": localization, "landmarks": [{"id": i,"x": float(p[0]),"y": float(p[1])} for i,p in enumerate(points)],
                    "geometry": geometry, "holistic": holistic,
                    "model": {"architecture": "MediaPipe Face Landmarker", "dataset": "mediapipe", "landmarks": 478,"initialization": "pretrained"},
                    "reliability": {"warnings": warnings,"geometry_error": {},"interpretation": "Точность новых ориентиров на независимой разметке в этой ветке ещё не измерена. Ошибки старой CNN к ним не относятся."},
                    "stages": [{"name": "Поиск лица и ориентиры", "detail": "MediaPipe; 478 точек и оценка ракурса", "milliseconds": round((before_regression-start)*1000,2),"status": "completed"},
                               {"name": "Визуальная оценка", "detail": ("ResNet-50 + личная регрессия; лицо и голова" if uses_head else "ResNet-50 + личная регрессия; фрагмент лица") if self.personal else "ResNet-50; 3 кадрирования × 2 отражения", "milliseconds": regression_ms,"status": "completed"}],
                    "total_milliseconds": elapsed}
        return {"localization": localization,"analysis": analysis}
