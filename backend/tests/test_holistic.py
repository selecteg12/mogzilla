"""API contracts and real-model integration; external weights may be absent in CI."""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import main
from backend.ml.holistic import HolisticAnalyzer, MODEL_DIR


def png():
    stream = io.BytesIO()
    Image.new("RGB",(300,350),(130,130,130)).save(stream,format="PNG")
    return stream.getvalue()


def test_holistic_api_requires_complete_region_and_supported_mime():
    client = TestClient(main.app)
    assert client.post("/api/holistic/analyze?x=1",content=png(),headers={"content-type":"image/png"}).status_code == 422
    assert client.post("/api/holistic/analyze",content=png(),headers={"content-type":"text/plain"}).status_code == 415
    assert client.post("/api/holistic/analyze",content=b"x"*(10*1024**2+1),headers={"content-type":"image/png"}).status_code == 413


def test_missing_research_weights_never_fall_back_to_old_score(tmp_path,monkeypatch):
    monkeypatch.setattr(main,"holistic_analyzer",HolisticAnalyzer(tmp_path))
    client=TestClient(main.app)
    assert client.get("/api/holistic/model").json()["ready"] is False
    response=client.post("/api/holistic/analyze",content=png(),headers={"content-type":"image/png"})
    assert response.status_code == 503
    assert "rating" not in response.json() and "holistic" not in response.json()


@pytest.fixture(scope="module")
def real_analyzer():
    if not (MODEL_DIR/"face_landmarker.task").is_file() or not (MODEL_DIR/"beauty_regressor.pt").is_file():
        pytest.skip("Downloaded research weights are not part of Git")
    analyzer=HolisticAnalyzer(personal=False)
    assert analyzer.status()["ready"]
    return analyzer


def test_non_face_image_has_no_landmarks_or_score(real_analyzer):
    result=real_analyzer.analyze(png())
    assert result["analysis"] is None
    assert result["localization"]["status"] == "not_found"


def test_real_upload_uses_visual_regressor_and_native_landmarks(real_analyzer,monkeypatch):
    sample=Path(__file__).resolve().parents[2]/"training/experiments/holistic/samples/sample-05.jpg"
    if not sample.is_file():
        pytest.skip("Local research sample is not distributed")
    monkeypatch.setattr(main,"holistic_analyzer",real_analyzer)
    client=TestClient(main.app)
    response=client.post("/api/holistic/analyze",content=sample.read_bytes(),headers={"content-type":"image/jpeg"})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    data=response.json()["analysis"]
    assert len(data["landmarks"]) == 478
    assert "rating" not in data
    result=data["holistic"]
    assert result["value"] == pytest.approx((result["raw_value"]-1)*2.5,abs=.006)
    assert result["validation"]["human_agreement_measured_here"] is False
    assert len(result["regions"]) == 8
    assert all("score" not in region for region in result["regions"])
    assert "data:image" not in str({k:v for k,v in data.items() if k not in {"visual_model_image","preprocessed_image"}})
    metrics={row["id"]:row for row in data["geometry"]["metrics"]}
    for key in ("eye_tilt_0","eye_tilt_1","chin_lip","visible_upper_middle"):
        assert metrics[key]["points"] and metrics[key]["formula"]


def test_multiple_faces_require_selection(real_analyzer):
    samples=Path(__file__).resolve().parents[2]/"training/experiments/holistic/samples"
    if not (samples/"sample-01.jpg").is_file():
        pytest.skip("Local research samples are not distributed")
    left=Image.open(samples/"sample-01.jpg").convert("RGB").resize((400,500))
    right=Image.open(samples/"sample-03.jpg").convert("RGB").resize((400,500))
    group=Image.new("RGB",(800,500)); group.paste(left,(0,0)); group.paste(right,(400,0))
    stream=io.BytesIO(); group.save(stream,format="PNG")
    result=real_analyzer.analyze(stream.getvalue())
    assert result["analysis"] is None
    assert result["localization"]["status"] == "multiple_faces"


def test_squinting_sample_does_not_receive_a_score(real_analyzer):
    sample=Path(__file__).resolve().parents[2]/"training/experiments/holistic/samples/sample-01.jpg"
    if not sample.is_file():
        pytest.skip("Local research sample is not distributed")
    with pytest.raises(ValueError,match="прищурены"):
        real_analyzer.analyze(sample.read_bytes())
