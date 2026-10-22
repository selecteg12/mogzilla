"""Explainable 2D measurements. No attractiveness, medical or body composition inference."""
import math
import numpy as np
from backend.geometry.topology import make_topology
from backend.geometry.descriptions import describe


def measure(points: np.ndarray, dataset: str) -> dict:
    t = make_topology(dataset)
    p = np.asarray(points,dtype=float)
    if p.shape != (t.count,2) or not np.isfinite(p).all():
        raise ValueError(f"Expected {t.count} finite (x, y) landmark pairs.")
    def distance(a,b):
        return float(np.linalg.norm(p[a]-p[b]))
    width = distance(*t.width)
    eye_centers = [p[list(e)].mean(0) for e in t.eyes]
    delta = eye_centers[1]-eye_centers[0]
    if width < 1e-6 or np.linalg.norm(delta) < 1e-6:
        raise ValueError("Landmarks are degenerate: face width or eye baseline is zero.")
    roll = math.atan2(delta[1],delta[0])
    c,s = math.cos(roll),math.sin(roll)
    rotation = np.array([[c,s],[-s,c]])
    origin = (eye_centers[0]+eye_centers[1])/2
    q = (p-origin)@rotation.T
    rows = []
    common = "Измерение в плоскости фотографии. Зависит от ракурса, мимики, кадрирования, перекрытий и ошибки точек. Идеальное значение не задаётся."
    def add(key,title,category,value,unit,ids,formula,details,limitation=common):
        rows.append({"id":key,"title":title,"description":describe(key),"category":category,"value":round(float(value),5) if value is not None else None,
                     "unit":unit,"point_ids":list(dict.fromkeys(ids)),"points":[{"id":i,"x":float(p[i,0]),"y":float(p[i,1])} for i in dict.fromkeys(ids)],
                     "formula":formula,"substitution":details,"limitation":limitation})
    tilts,aspects = [],[]
    for k,label in enumerate(("Слева на фото","Справа на фото")):
        inner,outer = t.corners[k]
        outward = q[outer]-q[inner]
        tilt = math.degrees(math.atan2(-outward[1],abs(outward[0])))
        tilts.append(tilt)
        add(f"eye_tilt_{k}",f"{label}: наклон глаза (canthal tilt)","Глаза",tilt,"°",list(t.eyes[0])+list(t.eyes[1]),
            f"θ = atan2(−(q{outer}.y − q{inner}.y), |q{outer}.x − q{inner}.x|) × 180/π",
            {"inner":inner,"outer":outer,"delta_outward":outward.tolist(),"roll_degrees":math.degrees(roll),
             "raw_image_angle_degrees":math.degrees(math.atan2(-(p[outer,1]-p[inner,1]),abs(p[outer,0]-p[inner,0]))),
             "aligned_points":[{"id":i,"x":float(q[i,0]),"y":float(q[i,1])} for i in [inner,outer]]},
            common+" Наклон головы убран по средним центрам контуров глаз. Положительное значение означает, что внешний уголок выше внутреннего. Ось y фото направлена вниз.")
        ew = distance(inner,outer)
        height = float(np.mean([distance(a,b) for a,b in t.lids[k]]))
        aspect = ew/height if height>1e-6 else None
        aspects.append(aspect)
        add(f"eye_aspect_{k}",f"{label}: пропорции глаза (W/H)","Глаза",aspect,"ratio",list(t.eyes[k]),
            f"||p{inner}−p{outer}|| / mean("+", ".join(f"||p{a}−p{b}||" for a,b in t.lids[k])+")",
            {"width_px":ew,"mean_visible_height_px":height},common+" Видимое раскрытие зависит от моргания и мимики.")
        brow = t.brows[k]
        brow_center = p[list(brow)].mean(0)
        gap = float(np.linalg.norm(brow_center-eye_centers[k]))/width
        add(f"brow_gap_{k}",f"{label}: расстояние бровь–глаз","Брови",gap,"ratio",list(brow)+list(t.eyes[k])+list(t.width),
            "||mean(brow points) − mean(eye points)|| / face contour width",
            {"brow_center":brow_center.tolist(),"eye_center":eye_centers[k].tolist(),"face_width_px":width})
        # Inner -> outer, with mirrored horizontal axis on the right side.
        bi,bo = (brow[-1],brow[0]) if k == 0 else (brow[0],brow[-1])
        bdelta = q[bo]-q[bi]
        angle = math.degrees(math.atan2(-bdelta[1],abs(bdelta[0])))
        add(f"brow_angle_{k}",f"{label}: наклон брови","Брови",angle,"°",list(brow)+list(t.eyes[0])+list(t.eyes[1]),
            f"atan2(−(q{bo}.y−q{bi}.y), |q{bo}.x−q{bi}.x|) × 180/π",
            {"inner":bi,"outer":bo,"aligned_delta":bdelta.tolist(),"roll_degrees":math.degrees(roll)})
        chord = q[bo]-q[bi]
        clen = float(np.linalg.norm(chord))
        deviation = float(np.max(np.abs(chord[0]*(q[list(brow),1]-q[bi,1])-chord[1]*(q[list(brow),0]-q[bi,0]))))/clen if clen>1e-6 else None
        add(f"brow_arch_{k}",f"{label}: изгиб брови","Брови",deviation/clen if deviation is not None else None,"ratio",list(brow),
            "max |cross(p_outer−p_inner, p_i−p_inner)| / ||p_outer−p_inner||²",
            {"chord_px":clen,"max_perpendicular_deviation_px":deviation},common+" Описание изгиба по редким точкам; густоту волос оно не измеряет.")
    inter = distance(t.corners[0][0],t.corners[1][0])
    add("intereye","Межуголковое расстояние / ширина контура","Глаза",inter/width,"ratio",[t.corners[0][0],t.corners[1][0],*t.width],
        "distance(inner eye corners) / face contour width",{"inner_distance_px":inter,"face_width_px":width})
    add("eye_tilt_difference","Разница наклона глаз","Глаза",abs(tilts[0]-tilts[1]),"°",list(t.eyes[0])+list(t.eyes[1]),
        "|θ_image_left − θ_image_right|",{"left_degrees":tilts[0],"right_degrees":tilts[1]})
    add("eye_aspect_difference","Разница пропорций глаз","Глаза",abs(aspects[0]-aspects[1]) if all(v is not None for v in aspects) else None,
        "ratio",list(t.eyes[0])+list(t.eyes[1]),"|aspect_image_left − aspect_image_right|",{"aspects":aspects})
    jaw = distance(*t.jaw)
    add("jaw_width","Ширина челюсти / лица (контурное приближение)","Нижняя часть лица",jaw/width,"ratio",[*t.jaw,*t.width],
        f"||p{t.jaw[0]}−p{t.jaw[1]}|| / ||p{t.width[0]}−p{t.width[1]}||",{"lower_width_px":jaw,"face_width_px":width},
        common+" Точки приближают видимый контур челюсти и не измеряют кость.")
    mid = distance(t.bridge,t.nose)
    lower = distance(t.nose,t.chin)
    add("lower_middle","Нижний / средний сегмент лица","Нижняя часть лица",lower/mid if mid>1e-6 else None,"ratio",[t.bridge,t.nose,t.chin],
        f"||p{t.nose}−p{t.chin}|| / ||p{t.bridge}−p{t.nose}||",{"lower_px":lower,"middle_px":mid},
        common+" Сегменты: переносица — основание носа — подбородок. Это не классические трети лица: линия роста волос не размечена.")
    height = distance(t.bridge,t.chin)
    add("width_height","Ширина контура / видимая высота","Пропорции",width/height if height>1e-6 else None,"ratio",[*t.width,t.bridge,t.chin],
        f"||p{t.width[0]}−p{t.width[1]}|| / ||p{t.bridge}−p{t.chin}||",{"width_px":width,"bridge_to_chin_px":height},
        common+" Высота начинается от переносицы, ширина берётся между верхними концами контура челюсти.")
    reflected = q.copy()
    reflected[:,0] *= -1
    deviations = [float(np.linalg.norm(reflected[a]-q[b])) for a,b in t.pairs]
    add("asymmetry","Асимметрия проекций ориентиров","Пропорции",float(np.mean(deviations))/width,"ratio",list(range(t.count)),
        "mean ||reflect_x(q_i) − q_j|| / face contour width; (i,j) are homologous pairs",
        {"axis_origin":origin.tolist(),"roll_degrees":math.degrees(roll),"pairs":[list(pair) for pair in t.pairs],
         "pair_errors_px":deviations,"face_width_px":width},common+" Ось отражения проходит через середину между глазами перпендикулярно их базовой линии. Это не внутренняя трёхмерная асимметрия.")
    nose_to_eye = [float(np.linalg.norm(q[t.nose]-q[list(e)].mean(0))) for e in t.eyes]
    imbalance = abs(nose_to_eye[0]-nose_to_eye[1])/max(sum(nose_to_eye),1e-6)
    warnings = []
    if abs(math.degrees(roll))>15:
        warnings.append("Сильный наклон головы в плоскости фото. Углы выровнены по линии глаз, перспективные искажения остаются.")
    if imbalance>0.15:
        warnings.append("Расстояния от носа до глаз отличаются: возможен поворот головы, асимметрия или ошибка точек. Перспектива может искажать измерения.")
    mean_eye_width = (distance(*t.corners[0])+distance(*t.corners[1]))/2
    add("eye_spacing","Межглазное расстояние (one-eye ratio)","Глаза",inter/mean_eye_width if mean_eye_width>1e-6 else None,"ratio",
        [*t.corners[0],*t.corners[1]],"inner-corner distance / mean(left eye width, right eye width)",
        {"inner_distance_px":inter,"mean_eye_width_px":mean_eye_width})
    mouth = (61,291) if dataset == "mediapipe" else (48,54) if dataset == "300w" else (76,82)
    alar = (98,327) if dataset == "mediapipe" else (31,35) if dataset == "300w" else (55,59)
    nose_width = distance(*alar)
    add("mouth_nose","Ширина рта / ширина носа","Пропорции",distance(*mouth)/nose_width if nose_width>1e-6 else None,"ratio",[*mouth,*alar],
        f"||p{mouth[0]}−p{mouth[1]}|| / ||p{alar[0]}−p{alar[1]}||",{"mouth_width_px":distance(*mouth),"nose_width_px":nose_width})
    unavailable = [{"term":"Facial thirds","reason":"Линия роста волос (trichion) не размечена."},
                   {"term":"Anatomical bigonial / bizygomatic ratio, fWHR","reason":"Gonion и zygion не размечены как анатомические ориентиры; контурные приближения им не эквивалентны."},
                   {"term":"Gonial angle, orbital vector, chin projection","reason":"Нужны подходящие профильные/3D-данные и дополнительные точки."},
                   {"term":"Upper eyelid exposure","reason":"Складка верхнего века не размечена."}]
    if dataset in {"wflw","mediapipe"}:
        left,right,lip = (468,473,0) if dataset == "mediapipe" else (96,97,79)
        ipd = distance(left,right)
        pupil_mid = (q[left]+q[right])/2
        lip_height = abs(float(q[lip,1]-pupil_mid[1]))
        add("midface_pupils","Midface ratio: зрачки — верхняя губа","Пропорции",ipd/lip_height if lip_height>1e-6 else None,"ratio",
            [left,right,lip,*t.eyes[0],*t.eyes[1]],f"||p{left}−p{right}|| / |q{lip}.y − (q{left}.y+q{right}.y)/2|",
            {"pupil_distance_px":ipd,"aligned_pupil_to_upper_lip_height_px":lip_height,"upper_lip_point":lip,"roll_degrees":math.degrees(roll)},
            common+" Выбрано конкретное определение, используемое в сообществе, а не универсальный анатомический стандарт. Положение зрачков зависит от направления взгляда.")
        if dataset == "mediapipe":
            upper = abs(float(q[168,1]-q[10,1]))
            middle = abs(float(q[2,1]-q[168,1]))
            add("visible_upper_middle","Видимая верхняя область / средний сегмент","Верхняя область",upper/middle if middle>1e-6 else None,"ratio",[10,168,2],
                "|q168.y − q10.y| / |q2.y − q168.y|",{"visible_upper_height_px":upper,"middle_height_px":middle},
                common+" Верхняя точка — видимый контур лба, не линия роста волос. Это не классические трети лица.")
            lip_height = distance(0,17)
            add("chin_lip","Подбородочный сегмент / высота губ","Нижняя часть лица",distance(17,152)/lip_height if lip_height>1e-6 else None,"ratio",[0,17,152],
                "||p17−p152|| / ||p0−p17||",{"lower_lip_to_chin_px":distance(17,152),"outer_lip_height_px":lip_height},
                common+" Нижняя губа, подбородок и внешняя высота губ в плоскости снимка; мимика меняет величину.")
            add("mouth_pupils","Ширина рта / межзрачковое расстояние","Пропорции",distance(61,291)/ipd if ipd>1e-6 else None,"ratio",[61,291,468,473],
                "||p61−p291|| / ||p468−p473||",{"mouth_width_px":distance(61,291),"pupil_distance_px":ipd})
            add("pupil_face","Межзрачковое расстояние / ширина контура","Глаза",ipd/width,"ratio",[468,473,234,454],
                "||p468−p473|| / ||p234−p454||",{"pupil_distance_px":ipd,"contour_width_px":width})
    else:
        unavailable.append({"term":"Midface ratio through pupil centers","reason":"В 300-W/AFW зрачки не размечены. Центры контуров глаз не подставляются вместо зрачков."})
    return {"metrics":rows,"unavailable":unavailable,"roll_degrees":math.degrees(roll),"axis":{"origin":origin.tolist(),"direction":[-s,c]},
            "pose_proxy":imbalance,"warnings":warnings,"paths":[list(path) for path in t.paths],
            "coordinate_convention":"Пиксели считаются с нуля; x — вправо, y — вниз. Слева/справа означает сторону отображаемого снимка.",
            "alignment":{"formula":"q = R(−roll) (p − eye_midpoint)","origin":origin.tolist(),"rotation":rotation.tolist()}}
