"""Explicit subjective geometry preferences, NOT a learned or objective attractiveness model."""
from typing import Literal

import numpy as np
from pydantic import BaseModel,Field,FiniteFloat,model_validator

from backend.geometry.metrics import measure
from backend.geometry.descriptions import describe
from backend.geometry.topology import make_topology

Criterion = Literal["tilt_mean","aspect_mean","spacing","midface_pupils","mouth_nose"]


class RatingRule(BaseModel):
    id: Criterion
    low: FiniteFloat
    high: FiniteFloat
    falloff: FiniteFloat = Field(gt=0,le=100)
    weight: FiniteFloat = Field(ge=0,le=10)

    @model_validator(mode="after")
    def valid_band(self):
        if self.low>self.high:
            raise ValueError("Нижняя граница должна быть не больше верхней.")
        return self


class RatingRequest(BaseModel):
    dataset: Literal["wflw","300w"]
    points: list[list[FiniteFloat]] = Field(min_length=68,max_length=98)
    rules: list[RatingRule] = Field(min_length=1,max_length=5)
    landmarks_reviewed: bool = False

    @model_validator(mode="after")
    def distinct_rules(self):
        if len({rule.id for rule in self.rules})!=len(self.rules):
            raise ValueError("Критерии не должны повторяться.")
        if any(len(point)!=2 for point in self.points):
            raise ValueError("Нужны пары координат x,y.")
        if len(self.points)!=(98 if self.dataset=="wflw" else 68):
            raise ValueError("Число точек должно соответствовать выбранной разметке.")
        return self


CATALOG = [
    {"id":"tilt_mean","title":"Наклон уголков глаз","unit":"°","metrics":["eye_tilt_0","eye_tilt_1"],"low":5.2,"high":8.5,"falloff":8.,"weight":1.},
    {"id":"aspect_mean","title":"Ширина / высота глаз","unit":"ratio","metrics":["eye_aspect_0","eye_aspect_1"],"low":2.8,"high":3.6,"falloff":2.,"weight":1.},
    {"id":"spacing","title":"Межглазное расстояние / ширина глаза","unit":"ratio","metrics":["eye_spacing"],"low":.95,"high":1.05,"falloff":.6,"weight":1.},
    {"id":"midface_pupils","title":"Midface ratio через зрачки","unit":"ratio","metrics":["midface_pupils"],"low":.93,"high":1.01,"falloff":.45,"weight":1.},
    {"id":"mouth_nose","title":"Ширина рта / ширина носа","unit":"ratio","metrics":["mouth_nose"],"low":1.38,"high":1.53,"falloff":.8,"weight":1.},
]


def settings():
    return {"criteria":[{**item,"description":describe(item["metrics"][0])} for item in CATALOG],"preset_name":"Редактируемый пример геометрических предпочтений",
            "source":"https://looksmax.org/threads/harmony-calculator-including-ideal-ratios-score-sheet.1264955/",
            "interpretation":"Это субъективный рейтинг привлекательности по выбранным пропорциям. Диапазоны — пример из отдельного форумного калькулятора с ограниченной применимостью, а не научная или общая норма. Веса и плавность штрафа выбраны для этого приложения и редактируются. Модель не обучалась на оценках привлекательности.",
            "formula":"s_i = 10 × max(0, 1 − max(low_i − x_i, 0, x_i − high_i) / falloff_i); S = Σ(w_i × s_i) / Σw_i",
            "coverage":"Только доступные фронтальные пропорции. Кожа, волосы, выражение, профиль и неразмеченные анатомические точки в рейтинг не входят."}


def criterion_score(value,rule):
    distance=max(rule.low-value,0.,value-rule.high)
    return 10*max(0.,1-distance/rule.falloff)


def annotations(points,dataset,criterion,geometry):
    """Endpoints are measured in the original photo; no invented ideal face."""
    t=make_topology(dataset)
    lines=[]
    def line(a,b,label,kind="measure"):
        lines.append({"a":np.asarray(a,dtype=float).tolist(),"b":np.asarray(b,dtype=float).tolist(),"label":label,"kind":kind})
    if criterion=="tilt_mean":
        direction=np.array([geometry["axis"]["direction"][1],-geometry["axis"]["direction"][0]])
        for inner,outer in t.corners:
            line(points[inner],points[outer],"Уголки глаза")
            length=float(np.dot(points[outer]-points[inner],direction))
            line(points[inner],points[inner]+direction*length,"Базовая линия после выравнивания","reference")
    elif criterion=="aspect_mean":
        for corners,lids in zip(t.corners,t.lids):
            line(points[corners[0]],points[corners[1]],"Ширина глаза")
            for a,b in lids:
                line(points[a],points[b],"Раскрытие век")
    elif criterion=="spacing":
        line(points[t.corners[0][0]],points[t.corners[1][0]],"Промежуток между глазами")
        for a,b in t.corners:
            line(points[a],points[b],"Ширина глаза")
    elif criterion=="mouth_nose":
        mouth=(48,54) if dataset=="300w" else (76,82)
        nose=(31,35) if dataset=="300w" else (55,59)
        line(points[mouth[0]],points[mouth[1]],"Ширина рта")
        line(points[nose[0]],points[nose[1]],"Ширина носа")
    elif criterion=="midface_pupils" and dataset=="wflw":
        mid=(points[96]+points[97])/2
        direction=np.asarray(geometry["axis"]["direction"])
        end=mid+direction*float(np.dot(points[79]-mid,direction))
        line(points[96],points[97],"Межзрачковое расстояние")
        line(mid,end,"Высота до верхней губы")
        line(end,points[79],"Проекция верхней губы","reference")
    return lines


def calculate(request: RatingRequest,metric_errors: dict | None = None):
    points=np.asarray(request.points,dtype=float)
    geometry=measure(points,request.dataset)
    rows={row["id"]:row for row in geometry["metrics"]}
    catalog={row["id"]:row for row in CATALOG}
    parts=[]; unavailable=[]
    for rule in request.rules:
        if rule.weight==0:
            continue
        definition=catalog[rule.id]
        ids=definition["metrics"]
        if not all(i in rows and rows[i]["value"] is not None for i in ids):
            unavailable.append({"id":rule.id,"reason":"Нужные точки отсутствуют или измерение вырождено."})
            continue
        value=float(np.mean([rows[i]["value"] for i in ids]))
        score=criterion_score(value,rule)
        deviation=max(rule.low-value,0.,value-rule.high)
        direction="below" if value<rule.low else "above" if value>rule.high else "within"
        error_values=[(metric_errors or {}).get(i,{}).get("mean_absolute_error") for i in ids]
        error=float(np.mean(error_values)) if all(e is not None for e in error_values) else None
        lower=upper=None
        if error is not None:
            lower=min(criterion_score(value-error,rule),criterion_score(value+error,rule))
            closest=min(max(value,rule.low),rule.high)
            best=min(max(closest,value-error),value+error)
            upper=criterion_score(best,rule)
        parts.append({"id":rule.id,"title":definition["title"],"unit":definition["unit"],"metric_ids":ids,
                      "description":describe(ids[0]),"annotations":annotations(points,request.dataset,rule.id,geometry),
                      "deviation":{"direction":direction,"distance":deviation,"deduction":10-score},
                      "observed":value,"low":rule.low,"high":rule.high,"falloff":rule.falloff,"weight":rule.weight,
                      "score":round(score,3),"weighted_points":score*rule.weight,"test_mae":error,
                      "sensitivity_low":lower,"sensitivity_high":upper,
                      "point_ids":sorted({p for i in ids for p in rows[i]["point_ids"]})})
    total=sum(part["weight"] for part in parts)
    if not total:
        return {"status":"unavailable","value":None,"message":"Не осталось доступных критериев с положительным весом.","unavailable":unavailable}
    for part in parts:
        part["contribution"]=part["weighted_points"]/total
        part["lost_contribution"]=(10-part["score"])*part["weight"]/total
        delta=part["deviation"]
        unit="°" if part["unit"]=="°" else " отн."
        part["summary"]=("В выбранном диапазоне." if delta["direction"]=="within" else
                         f"На {delta['distance']:.2f}{unit} {'ниже' if delta['direction']=='below' else 'выше'} выбранного диапазона.")
    result={"status":"calculated","value":round(sum(part["weighted_points"] for part in parts)/total,2),
            "scale":[0,10],"parts":parts,"unavailable":unavailable,"used_criteria":len(parts),
            "formula":settings()["formula"],"kind":"subjective_rule_based_attractiveness_from_geometry",
            "interpretation":settings()["interpretation"],"coverage":settings()["coverage"],
            "landmarks_reviewed":request.landmarks_reviewed,"provisional":not request.landmarks_reviewed,
            "review_message":"Проверьте совпадение точек с лицом. До проверки балл предварительный." if not request.landmarks_reviewed else "Пользователь подтвердил положение точек. Это не измерение точности модели.",
            "sensitivity":None}
    if all(part["test_mae"] is not None for part in parts):
        result["sensitivity"]={"low":round(sum(p["sensitivity_low"]*p["weight"] for p in parts)/total,2),
                               "high":round(sum(p["sensitivity_high"]*p["weight"] for p in parts)/total,2),
                               "interpretation":"Что получится при сдвиге каждого измерения в пределах его средней ошибки теста. Это анализ чувствительности, не доверительный интервал и не вероятность."}
    return result
