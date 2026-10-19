"""A constructed geometry target. It is explicitly NOT a face model prediction."""
import math

import numpy as np

from backend.geometry.metrics import measure
from backend.geometry.rating import CATALOG,RatingRequest,calculate,settings


def ideal_reference():
    rules={rule["id"]:rule for rule in CATALOG}
    middle=lambda key:(rules[key]["low"]+rules[key]["high"])/2
    tilt=math.radians(middle("tilt_mean"))
    horizontal=60.
    rise=horizontal*math.tan(tilt)
    eye_width=math.hypot(horizontal,rise)
    gap=eye_width*middle("spacing")
    left_outer,left_inner=180.,240.
    right_inner=left_inner+gap
    right_outer=right_inner+horizontal
    cx=(left_outer+right_outer)/2
    pupil_y=184-rise/2
    pupil_distance=horizontal+gap
    lip_y=pupil_y+pupil_distance/middle("midface_pupils")
    p=np.zeros((98,2),dtype=float)
    for i,t in enumerate(np.linspace(0,math.pi,33)):
        p[i]=[cx-160*math.cos(t),142+250*math.sin(t)]
    for start,x0 in [(33,left_outer),(42,right_inner)]:
        p[start:start+5]=[[x0+j*15,148-7*math.sin(j*math.pi/4)] for j in range(5)]
        p[start+5:start+9]=[[x0+(4-j)*15,153-5*math.sin((4-j)*math.pi/4)] for j in range(4)]
    p[51:55]=[[cx,190],[cx,215],[cx,240],[cx,260]]
    p[55:60]=[[cx-20,271],[cx-10,276],[cx,278],[cx+10,276],[cx+20,271]]
    eye_height=eye_width/middle("aspect_mean")
    for start,x0,sign in [(60,left_outer,1),(68,right_inner,-1)]:
        for j,(fraction,vertical) in enumerate([(0,0),(.25,-1),(.5,-1),(.75,-1),(1,0),(.75,1),(.5,1),(.25,1)]):
            baseline=184-rise+fraction*rise if sign==1 else 184-fraction*rise
            p[start+j]=[x0+horizontal*fraction,baseline+vertical*eye_height/2]
    mouth_width=40*middle("mouth_nose")
    for j,t in enumerate(np.linspace(0,2*math.pi,12,endpoint=False)):
        p[76+j]=[cx-mouth_width/2*math.cos(t),lip_y+8-8*math.sin(t)]
    for j,t in enumerate(np.linspace(0,2*math.pi,8,endpoint=False)):
        p[88+j]=[cx-mouth_width*.4*math.cos(t),lip_y+8-3*math.sin(t)]
    p[96]=[(left_outer+left_inner)/2,pupil_y]
    p[97]=[(right_inner+right_outer)/2,pupil_y]
    rating=calculate(RatingRequest(dataset="wflw",points=p.tolist(),rules=CATALOG))
    geometry=measure(p,"wflw")
    return {"kind":"constructed_geometry_not_prediction","width":550,"height":470,
            "points":[{"id":i,"x":float(point[0]),"y":float(point[1])} for i,point in enumerate(p)],
            "paths":geometry["paths"],"rating":rating,"criteria":settings()["criteria"],
            "interpretation":"Геометрическая схема построена по серединам заданных диапазонов. Это не фотография и не предсказание CNN. Все пять критериев попадают в диапазоны на 10/10 по текущей формуле приложения."}
