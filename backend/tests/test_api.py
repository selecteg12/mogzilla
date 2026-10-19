import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import main
from backend.ml.inference import Predictor
from backend.ml.regions import RegionFinder


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(main,"predictor",Predictor(tmp_path/"absent.pt"))
    monkeypatch.setattr(main,"region_finder",RegionFinder(tmp_path/"absent-region.pt"))
    return TestClient(main.app)


def picture(format="PNG"):
    image = Image.new("RGB",(200,220),(130,120,115))
    stream = io.BytesIO(); image.save(stream,format=format)
    return stream.getvalue()


def test_preparation_works_without_model_and_inference_fails_closed(client):
    response = client.post("/api/prepare?x=20&y=20&width=160&height=180",content=picture(),headers={"Content-Type":"image/png"})
    assert response.status_code==200
    data = response.json()
    assert len(data["stages"])==3
    assert "landmarks" not in data
    assert data["transform"]["size"]==96
    assert response.headers["Cache-Control"]=="no-store"
    response = client.post("/api/analyze?x=20&y=20&width=160&height=180",content=picture(),headers={"Content-Type":"image/png"})
    assert response.status_code==503
    assert not client.get("/api/model").json()["ready"]


@pytest.mark.parametrize("query",["x=-1&y=0&width=50&height=50","x=180&y=0&width=50&height=50","x=0&y=0&width=12&height=50","x=nan&y=0&width=50&height=50"])
def test_invalid_regions_are_rejected(client,query):
    assert client.post("/api/prepare?"+query,content=picture(),headers={"Content-Type":"image/png"}).status_code==422


def test_invalid_body_and_mime(client):
    url = "/api/prepare?x=0&y=0&width=50&height=50"
    assert client.post(url,content=b"not an image",headers={"Content-Type":"image/png"}).status_code==422
    assert client.post(url,content=picture("GIF"),headers={"Content-Type":"image/png"}).status_code==422
    assert client.post(url,content=picture(),headers={"Content-Type":"text/plain"}).status_code==415
    assert client.post(url,content=b"a"*(10*1024**2+1),headers={"Content-Type":"image/jpeg"}).status_code==413


def test_exif_is_applied_once():
    from backend.preprocessing.images import decode_image
    image = Image.new("RGB",(100,60))
    exif = Image.Exif(); exif[274] = 6
    stream = io.BytesIO(); image.save(stream,format="JPEG",exif=exif)
    assert decode_image(stream.getvalue()).shape==(100,60,3)


def test_localization_without_weights_has_no_fake_points(client):
    result=client.post("/api/localize",content=picture(),headers={"Content-Type":"image/png"})
    assert result.status_code==200
    assert result.json()["analysis"] is None
    assert result.json()["localization"]["status"]=="unavailable"


def test_bad_framing_cannot_reach_landmark_model(client,monkeypatch):
    monkeypatch.setattr(main.region_finder,"score",lambda *args: .01)
    result=client.post("/api/analyze?x=0&y=0&width=200&height=220",content=picture(),headers={"Content-Type":"image/png"})
    assert result.status_code==422
    assert "крупный кадр лица" in result.json()["detail"]


def test_partial_face_cannot_produce_points_or_rating(client,monkeypatch):
    monkeypatch.setattr(main.region_finder,"score",lambda *args: .99)
    monkeypatch.setattr(main.region_finder,"verify",lambda *args: {"complete":False})
    def forbidden(*args):
        pytest.fail("An incomplete face must not reach landmark inference or rating.")
    monkeypatch.setattr(main.predictor,"predict",forbidden)
    result=client.post("/api/analyze?x=20&y=20&width=120&height=120",content=picture(),headers={"Content-Type":"image/png"})
    assert result.status_code==422
    assert "Рейтинг" in result.json()["detail"]
    assert "landmarks" not in result.json() and "rating" not in result.json()


def test_failed_automatic_framing_abstains_without_points_or_rating(client,monkeypatch):
    monkeypatch.setattr(main.region_finder,"find",lambda *args: {"status":"found","candidates":[{"box":[20,20,120,120],"score":.99}]})
    monkeypatch.setattr(main.region_finder,"verify",lambda *args: {"complete":False})
    def forbidden(*args):
        pytest.fail("Rejected automatic framing must not reach landmark inference.")
    monkeypatch.setattr(main.predictor,"predict",forbidden)
    result=client.post("/api/localize",content=picture(),headers={"Content-Type":"image/png"})
    assert result.status_code==200
    assert result.json()["analysis"] is None
    assert result.json()["localization"]["status"]=="not_found"


def test_unstable_landmarks_cannot_produce_rating_even_for_manual_region(client,monkeypatch):
    monkeypatch.setattr(main.predictor,"predict",lambda *args: {"flip_disagreement_nme":.2})
    result=client.post("/api/analyze?x=20&y=20&width=120&height=120",content=picture(),headers={"Content-Type":"image/png"})
    assert result.status_code==422
    assert "нестабильны" in result.json()["detail"]
    assert "rating" not in result.json()


def test_rating_api_recomputes_geometry_and_labels_provisional_score(client):
    from backend.tests.test_geometry import face
    rule={"id":"spacing","low":.95,"high":1.05,"falloff":.6,"weight":1}
    payload={"dataset":"300w","points":face().tolist(),"rules":[rule]}
    initial=client.post("/api/rating",json=payload).json()
    assert initial["value"]==10
    assert initial["provisional"] is True
    assert initial["parts"][0]["annotations"]
    payload["landmarks_reviewed"]=True
    response=client.post("/api/rating",json=payload)
    assert response.status_code==200
    assert response.json()["value"]==10
    assert response.json()["provisional"] is False
    payload["points"]=[[0,0]]*68
    assert client.post("/api/rating",json=payload).status_code==422


def test_reference_page_has_target_values_even_without_a_trained_model(client):
    response=client.get("/api/reference-values")
    assert response.status_code==200
    assert response.json()["reference"]["rating"]["value"]==10
    assert response.json()["reference"]["kind"]=="constructed_geometry_not_prediction"
    # Historical gallery scores must not be presented as predictions of new/missing weights.
    assert response.json()["gallery"]["examples"]==[]
    assert response.headers["Cache-Control"]=="no-store"


@pytest.mark.parametrize("example_id",[0,4,-1,100])
def test_gallery_image_route_does_not_expose_arbitrary_files(client,example_id):
    assert client.get(f"/api/reference-values/photo/{example_id}").status_code==404


def test_old_gallery_revision_cannot_silently_serve_a_new_photo(client):
    assert client.get("/api/reference-values/photo/1?v=obsolete-gallery").status_code==404
