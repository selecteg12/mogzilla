"""Local REST boundary between React and Python. Без внешних AI API or upload storage."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI,HTTPException,Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.geometry.metrics import measure
from backend.geometry.rating import RatingRequest,calculate,settings
from backend.geometry.reference import ideal_reference
from backend.geometry.reference_gallery import gallery_is_current
from backend.ml.inference import Predictor,ModelUnavailable
from backend.ml.regions import RegionFinder
from backend.ml.holistic import HolisticAnalyzer
from backend.preprocessing.images import MAX_BYTES,crop_image,decode_image,encode_image,image_quality

ROOT = Path(__file__).resolve().parents[2]
predictor = Predictor()
region_finder = RegionFinder()
holistic_analyzer = HolisticAnalyzer()
app = FastAPI(title="mogzilla",version="0.1.0",description="Local visual portrait scoring and explainable 2D geometry.")


@app.get("/api/holistic/model")
def holistic_model_status():
    return holistic_analyzer.status()


@app.get("/api/holistic/method")
def holistic_method():
    report = ROOT/"reports/holistic/evaluation.json"
    evaluation = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
    if evaluation:
        evaluation = {k:v for k,v in evaluation.items() if k!="rows"}
    return {"model":holistic_analyzer.status(),"evaluation":evaluation}


@app.post("/api/holistic/analyze")
async def holistic_analyze(request: Request,x: float | None = None,y: float | None = None,width: float | None = None,height: float | None = None):
    coordinates = [x,y,width,height]
    if any(v is not None for v in coordinates) and not all(v is not None for v in coordinates):
        raise HTTPException(422,"Передайте все четыре координаты области лица.")
    data = await read_photo(request)
    try:
        return await run_in_threadpool(holistic_analyzer.analyze,data,coordinates if x is not None else None)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503,str(exc)) from exc


@app.middleware("http")
async def headers(request: Request,call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    return response


@app.get("/api/health")
def health():
    return {"status":"ok","service":"FaceMetrics AI","external_ai":False,"photo_storage":False}


@app.get("/api/model")
def model_status():
    return {**predictor.status(),"region_finder":region_finder.status()}


@app.get("/api/rating/settings")
def rating_settings():
    return settings()


@app.get("/api/reference-values")
def reference_values():
    gallery_path=ROOT/"reports/reference-values/examples.json"
    gallery={"examples":[],"message":"Проверенные фотографии-примеры ещё не подготовлены."}
    if gallery_path.is_file():
        gallery=json.loads(gallery_path.read_text(encoding="utf-8"))
        matches=gallery_is_current(gallery,predictor.path,region_finder.path)
        if not matches:
            gallery={"examples":[],"message":"Поиск лица, модель точек или шкала изменились. Фотографии-примеры пересчитываются для текущего анализа."}
    return {"reference":ideal_reference(),"gallery":gallery}


@app.get("/api/reference-values/photo/{example_id}")
def reference_photo(example_id: int,v: str | None = None):
    # Fixed gallery only: dataset folders and user photos are never served.
    if example_id not in {1,2,3}:
        raise HTTPException(404,"Фотография-пример не найдена.")
    if v is not None:
        gallery_path=ROOT/"reports/reference-values/examples.json"
        gallery=json.loads(gallery_path.read_text(encoding="utf-8")) if gallery_path.is_file() else {}
        version=gallery.get("generated_at","").replace(":","").replace("+","")
        if not version or v!=version:
            raise HTTPException(404,"Галерея обновилась. Обновите страницу, чтобы увидеть актуальные фото и баллы.")
    path=ROOT/f"reports/reference-values/example-{example_id}.jpg"
    if not path.is_file():
        raise HTTPException(404,"Фотография-пример ещё не подготовлена.")
    return FileResponse(path,media_type="image/jpeg")


@app.post("/api/rating")
def rate_geometry(payload: RatingRequest):
    metadata=predictor.status().get("metadata") or {}
    errors=metadata.get("evaluation",{}).get("geometry_error",{}) if metadata.get("dataset")==payload.dataset else {}
    try:
        return calculate(payload,errors)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc


@app.get("/api/research")
def research():
    report = ROOT/"reports/run/evaluation.json"
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    return {"status":"awaiting_training","comparison":[],"test":None,
            "message":"Результаты эксперимента ещё не получены. Метрики и графики появятся после реального обучения и независимой проверки."}


@app.get("/api/research/figure/{name}")
def research_figure(name: str):
    # Deliberate allowlist. No path-based serving of dataset or user images.
    figures = {"ced":"reports/run/ced.png","errors":"reports/run/prediction_errors.png",
               "coordinates-loss":"training/experiments/wflw-coordinates/loss.png",
               "heatmaps-loss":"training/experiments/wflw-heatmaps/loss.png"}
    if name not in figures or not (ROOT/figures[name]).is_file():
        raise HTTPException(404,"Figure has not been generated.")
    return FileResponse(ROOT/figures[name])


async def read_photo(request: Request):
    if request.headers.get("content-type","").split(";")[0] not in {"image/jpeg","image/png","image/webp"}:
        raise HTTPException(415,"Передайте JPEG, PNG или WebP в теле запроса.")
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length)>MAX_BYTES):
        raise HTTPException(413,"Изображение превышает лимит 10 МБ.")
    data = bytearray()
    async for chunk in request.stream():
        if len(data)+len(chunk)>MAX_BYTES:
            raise HTTPException(413,"Изображение превышает лимит 10 МБ.")
        data.extend(chunk)
    return bytes(data)


def process(data: bytes,box: list[float],analyze: bool,localization: dict | None = None):
    start = time.perf_counter()
    stages = []
    def stage(name,detail,elapsed):
        stages.append({"name":name,"detail":detail,"milliseconds":round(elapsed*1000,2),"status":"completed"})
    try:
        rgb = decode_image(data)
        stage("Загрузка изображения",f"{len(data)} байт; декодировано {rgb.shape[1]} × {rgb.shape[0]} RGB; применена ориентация EXIF",time.perf_counter()-start)
        origin="Область, предложенная собственной CNN" if localization else "Область, выделенная вручную"
        framing_detail=" Поиск по масштабам, регрессия границ и повторная проверка полного лица." if localization and localization.get("method")=="complete_face_box_v1" else ""
        stage("Выделение области лица",f"{origin} в пикселях ориентированного снимка: {box}.{framing_detail}",0)
        before = time.perf_counter()
        crop,transform = crop_image(rgb,box)
        quality = image_quality(crop)
        stage("Предобработка изображения","Квадратное кадрирование с отступом 12%, отражение на краях, размер 96 × 96, RGB / 127.5 − 1",time.perf_counter()-before)
        response = {"source":{"width":rgb.shape[1],"height":rgb.shape[0],"bytes":len(data)},"box":box,
                    "preprocessed_image":encode_image(crop),"transform":transform,"quality":quality,"stages":stages}
        if not analyze:
            return response
        region_score=region_finder.score(rgb,box)
        if region_score is not None and region_score<.5:
            raise ValueError("В выбранной области модель не распознала крупный кадр лица. Выделите только лицо: брови, боковой контур и подбородок. Полный рост и весь кадр не подходят.")
        framing=region_finder.verify(rgb,box)
        if framing is not None and not framing["complete"]:
            raise ValueError("Не удалось подтвердить целое лицо в этой области. В кадре должны помещаться оба глаза, нос, рот и подбородок. Рейтинг для непроверенной рамки не рассчитан.")
        response["localization"]=localization or {"status":"manual","score":region_score,"score_kind":"uncalibrated_crop_classifier"}
        if framing is not None:
            response["localization"]["framing_check"]=framing
        before = time.perf_counter()
        prediction = predictor.predict(crop,transform)
        if prediction["flip_disagreement_nme"]>.12:
            raise ValueError("Точки лица нестабильны при отражении снимка. Рейтинг не рассчитан: попробуйте чёткую фотографию анфас без сильного наклона головы.")
        metadata = prediction["metadata"]
        architecture = "Регрессия координат" if metadata["architecture"] == "coordinates" else "Карты ориентиров"
        stage("Запуск нейросети",f"{architecture}: CNN; {metadata['parameters']} обучаемых параметров; обучение со случайной инициализацией. Прямой проход исходного и отражённого снимка.",time.perf_counter()-before)
        points = prediction["points"]
        response["landmarks"] = [{"id":i,"x":float(p[0]),"y":float(p[1])} for i,p in enumerate(points)]
        stage("Локализация ориентиров",f"{len(points)} предсказанных координат переведены в пиксели исходного снимка",0)
        before = time.perf_counter()
        geometry = measure(points,metadata["dataset"])
        response["geometry"] = geometry
        stage("Геометрические измерения",f"{len(geometry['metrics'])} двумерных измерений по явным формулам",time.perf_counter()-before)
        warnings = quality["warnings"]+geometry["warnings"]
        if min(box[2:])<96:
            warnings.append("Исходная область лица меньше 96 пикселей. Увеличение размера не восстанавливает детали.")
        if prediction["flip_disagreement_nme"]>.05:
            warnings.append("Предсказания нестабильны при горизонтальном отражении. Проверьте положение точек перед интерпретацией измерений.")
        if metadata["test"]["nme"]>.08:
            warnings.append("Ошибка ориентиров на тесте велика. Ошибка модели может превышать малые углы и различия.")
        response["reliability"] = {"warnings":warnings,"flip_disagreement_nme":prediction["flip_disagreement_nme"],
            "validation":metadata["validation"],"test":metadata["test"],"geometry_error":metadata["evaluation"].get("geometry_error",{}),
            "confidence_probability":None,"interpretation":"Уверенность для отдельного фото не откалибрована. Ошибка теста — средняя по выборке, расхождение при отражении — проверка стабильности. Собственная CNN предлагает область лица, но может ошибаться. Проверьте совпадение точек с реальным лицом перед рейтингом."}
        response["model"] = metadata
        response["rating_settings"]=settings()
        response["rating"]=calculate(RatingRequest(dataset=metadata["dataset"],points=points.tolist(),
            rules=settings()["criteria"]),response["reliability"]["geometry_error"])
        if "heatmaps" in prediction:
            maps = prediction["heatmaps"]
            average = maps.mean(0)
            intensity = np.uint8(255*(average-average.min())/max(float(np.ptp(average)),1e-12))
            color = cv2.cvtColor(cv2.applyColorMap(intensity,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
            response["heatmap_image"] = encode_image(color)
            response["heatmap_description"] = "Среднее фактических пространственных softmax-выходов по ориентирам. Это визуализация, не карта внимания или уверенности. Координаты получены soft-argmax каждой отдельной карты."
        stage("Результаты",f"Фактические предсказания модели и расчёты; фото не сохранено",0)
        response["total_milliseconds"] = round((time.perf_counter()-start)*1000,2)
        return response
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503,str(exc)) from exc


@app.post("/api/prepare")
async def prepare_photo(request: Request,x: float,y: float,width: float,height: float):
    data = await read_photo(request)
    return await run_in_threadpool(process,data,[x,y,width,height],False)


@app.post("/api/analyze")
async def analyze_photo(request: Request,x: float,y: float,width: float,height: float):
    data = await read_photo(request)
    return await run_in_threadpool(process,data,[x,y,width,height],True)


def localize(data: bytes):
    try:
        found=region_finder.find(decode_image(data))
        if found["status"]!="found":
            return {"localization":found,"analysis":None}
        # Crop classification alone has false positives. Reject candidates whose
        # landmark predictions disagree strongly after a horizontal reflection.
        rgb=decode_image(data)
        for index,candidate in enumerate(found["candidates"]):
            framing=region_finder.verify(rgb,candidate["box"])
            candidate["framing_check"]=framing
            if framing is not None and not framing["complete"]:
                continue
            prediction=predictor.predict(*crop_image(rgb,candidate["box"]))
            candidate["flip_disagreement_nme"]=prediction["flip_disagreement_nme"]
            if prediction["flip_disagreement_nme"]<=.12:
                localization={**found,"selected":index,"score":candidate["score"],"stability_limit":.12}
                return {"localization":localization,"analysis":process(data,candidate["box"],True,localization)}
        return {"localization":{"status":"not_found","candidates":[],"message":"Проверка полного лица или устойчивости точек не пройдена. Загрузите чёткое фото анфас или выделите одно лицо вручную."},"analysis":None}
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503,str(exc)) from exc


@app.post("/api/localize")
async def localize_photo(request: Request):
    return await run_in_threadpool(localize,await read_photo(request))


# A built React app is served by the same Python origin. Vite proxies /api during development.
DIST = ROOT/"frontend/dist"
if DIST.is_dir():
    app.mount("/assets",StaticFiles(directory=DIST/"assets"),name="assets")


@app.get("/")
def index():
    if (DIST/"index.html").is_file():
        return FileResponse(DIST/"index.html")
    return JSONResponse({"message":"Build the frontend, or use Vite at http://localhost:5173. API docs: /docs"})
