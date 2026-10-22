"""Find high predictions using fixed preprocessing, without changing the model.

Exploratory selection only, not validation. Originals are read-only; portraits,
rankings and previews remain in ignored local research output directories.
"""
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/qa/holistic-near-ten"


def portrait_bytes(original: Image.Image, points: np.ndarray) -> bytes:
    lo,hi=points.min(0),points.max(0)
    w,h=hi-lo
    bounds=(max(0,int(lo[0]-.2*w)),max(0,int(lo[1]-.4*h)),
            min(original.width,int(hi[0]+.2*w)),min(original.height,int(hi[1]+.15*h)))
    out=io.BytesIO()
    original.crop(bounds).save(out,format="JPEG",quality=95)
    return out.getvalue()


def scan(prepared: Path, count: int, keep: int, url: str):
    started=time.perf_counter()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    image_dir=OUTPUT/"photos"
    image_dir.mkdir(exist_ok=True)
    metadata=json.loads((prepared/"dataset.json").read_text(encoding="utf-8"))
    records=[json.loads(line) for line in (prepared/"manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    records=[r for r in records if r["split"]=="test" and np.ptp(np.array(r["points"]),axis=0).min()>=110 and not r.get("out_of_frame_points")]
    records.sort(key=lambda r:hashlib.sha256((r["image"]+str(r["points"][0])).encode()).hexdigest())
    unique=[]
    seen=set()
    for row in records:
        key=row.get("image_sha256",row["image"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    records=unique[:count]
    rows=[]
    print(f"Scanning {len(records)} distinct source images; fixed portrait selection, model unchanged",flush=True)
    with httpx.Client(base_url=url,timeout=60) as client:
        status=client.get("/api/holistic/model").raise_for_status().json()
        if not status["ready"]:
            raise RuntimeError(status["reason"])
        for i,record in enumerate(records):
            with Image.open(Path(metadata["root"])/record["image"]) as source:
                data=portrait_bytes(source.convert("RGB"),np.array(record["points"]))
            response=client.post("/api/holistic/analyze",content=data,headers={"content-type":"image/jpeg"})
            row={"index":i+1,"source_image":record["image"],"input_sha256":hashlib.sha256(data).hexdigest()}
            if response.status_code==422:
                row.update(status="abstained",reason=response.json()["detail"])
            else:
                response.raise_for_status()
                body=response.json()
                analysis=body["analysis"]
                if analysis is None:
                    row.update(status="abstained",reason=body["localization"]["message"])
                else:
                    rating=analysis["holistic"]
                    path=image_dir/f"candidate-{i+1:03d}.jpg"
                    path.write_bytes(data)
                    row.update(status="calculated",value=rating["value"],raw_value=rating["raw_value"],
                               stability=rating["stability"],warnings=analysis["quality"]["warnings"],
                               pose=analysis["quality"]["pose"],path=str(path))
            rows.append(row)
            if (i+1)%25==0 or i+1==len(records):
                accepted=[r for r in rows if r["status"]=="calculated"]
                top=sorted(accepted,key=lambda r:-r["value"])[:3]
                print(f"{i+1}/{len(records)}; accepted={len(accepted)}; top={[r['value'] for r in top]}",flush=True)
    accepted=sorted([r for r in rows if r["status"]=="calculated"],key=lambda r:-r["value"])
    selected=accepted[:keep]
    report={"purpose":"exploratory high-score example selection; not a validation set", "model":status,
            "preprocessing":"same fixed GT-selected portrait as technical audit; default API crop and flip averaging; no optimization or model changes",
            "scanned":len(rows),"accepted":len(accepted),"abstained":len(rows)-len(accepted),
            "elapsed_seconds":round(time.perf_counter()-started,2),"selected":selected,
            "stable_high_examples":[r for r in accepted if r["stability"]["status"]=="stable_in_this_check"][:keep],
            "human_ratings":None,"rows":rows}
    (OUTPUT/"search.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    make_preview(selected,report)
    print(json.dumps({k:report[k] for k in ["scanned","accepted","abstained","elapsed_seconds","selected"]},ensure_ascii=False,indent=2))


def make_preview(selected: list[dict], report: dict):
    checks_path=OUTPUT/"browser-checks.json"
    browser_checks=json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.is_file() else []
    checks={int(row["id"]):row for row in browser_checks}
    font_path=Path("C:/Windows/Fonts/segoeui.ttf")
    font=lambda size:ImageFont.truetype(str(font_path),size)
    columns=3
    rows=(len(selected)+columns-1)//columns
    width,cell_w,cell_h=1200,380,500
    canvas=Image.new("RGB",(width,165+rows*cell_h),(17,18,25))
    draw=ImageDraw.Draw(canvas)
    draw.text((28,22),"Фото с высокими прогнозами модели",font=font(32),fill="#ece7f7")
    draw.text((28,72),f"Проверено {report['scanned']} снимков; пригодны {report['accepted']}. Отобраны по итоговому баллу.",font=font(19),fill="#a9a2ba")
    draw.text((28,108),"Проверены загрузкой в браузере. Оценки людей для этих фотографий неизвестны." if checks else "Это примеры поведения модели. Оценки людей для этих фотографий неизвестны.",font=font(19),fill="#c9b8e4")
    for i,row in enumerate(selected):
        x,y=20+(i%columns)*400,155+(i//columns)*cell_h
        draw.rounded_rectangle((x,y,x+cell_w,y+cell_h-15),radius=13,fill="#1c1d27",outline="#393144")
        photo=Image.open(row["path"]).convert("RGB")
        photo.thumbnail((cell_w-24,305),Image.Resampling.LANCZOS)
        canvas.paste(photo,(x+(cell_w-photo.width)//2,y+12+(305-photo.height)//2))
        check=checks.get(row["index"])
        score=check["score"] if check and "score" in check else f"{row['value']:.2f}"
        draw.text((x+18,y+333),f"{i+1}.  {score} / 10",font=font(27),fill="#d3b5ff")
        s=row["stability"]
        interval=check["stability"].split(' / ')[0] if check and "score" in check else f"{s['low']:.2f}–{s['high']:.2f}"
        draw.text((x+18,y+376),f"При другой рамке: {interval}",font=font(18),fill="#b4abc3")
        warnings=check.get("warnings",[]) if check else row["warnings"]
        label="Есть технические замечания" if warnings else "Нет замечаний в этом прогоне"
        draw.text((x+18,y+410),label,font=font(18),fill="#e3bf80" if warnings else "#98cfbb")
        draw.text((x+18,y+443),f"Прямой API исходного JPEG: {row['value']:.2f}",font=font(16),fill="#9f95b0")
    canvas.save(OUTPUT/"near-ten.png")
    text="Примеры, выбранные по высоким прогнозам модели, а не по известным человеческим оценкам.\n"
    text+=f"Проверено {report['scanned']} снимков; принято {report['accepted']}; отказов {report['abstained']}.\n"
    text+="Основной балл: (исходный прогноз по шкале 1–5 − 1) × 2.5.\nДиапазон при кадрировании — техническая чувствительность, не доверительный интервал.\nПри загрузке браузер повторно сжимает JPEG, поэтому итог может немного измениться.\nЗначения интерфейса округлены до одного знака, в том числе 10.0 — это округление.\nОтдельные измерения не складываются в общий прогноз ResNet.\n\n"
    with zipfile.ZipFile(OUTPUT/"near-ten-photos.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for i,row in enumerate(selected,1):
            name=f"{i:02d}-score-{row['value']:.2f}.jpg"
            archive.write(row["path"],name)
            check=checks.get(row["index"])
            text+=f"{name}: API={row['value']}; raw={row['raw_value']}; рамка API={row['stability']['low']}–{row['stability']['high']}; замечания API={'; '.join(row['warnings']) or 'нет'}.\n"
            if check:
                text+=f"  В браузере: {check.get('score',check.get('error'))}; рамка={check.get('stability','—')}; замечания={check.get('warnings',[])}\n"
        archive.write(OUTPUT/"near-ten.png","near-ten.png")
        if checks_path.is_file():
            archive.write(checks_path,"browser-checks.json")
        archive.writestr("README.txt",text)
    (OUTPUT/"README.txt").write_text(text,encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared",type=Path)
    parser.add_argument("--render-only",action="store_true")
    parser.add_argument("--count",type=int,default=300)
    parser.add_argument("--keep",type=int,default=6)
    parser.add_argument("--url",default="http://127.0.0.1:8001")
    args=parser.parse_args()
    if args.render_only:
        report=json.loads((OUTPUT/"search.json").read_text(encoding="utf-8"))
        make_preview(report["selected"],report)
    elif args.prepared is None:
        parser.error("--prepared is required for a new scan")
    else:
        scan(args.prepared,args.count,args.keep,args.url)
