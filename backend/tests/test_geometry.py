import math
import numpy as np
import pytest

from backend.geometry.metrics import measure
from backend.geometry.topology import make_topology
from backend.preprocessing.images import crop_image,image_points,normalized_points


def face():
    # Synthetic geometry fixture ONLY; never used as a model output or dataset.
    points = np.zeros((68,2),dtype=float)
    points[:17] = [[50-40*math.cos(i*math.pi/16),30+60*math.sin(i*math.pi/16)] for i in range(17)]
    points[17:22] = [[20,24],[25,21],[30,20],[35,21],[40,24]]
    points[22:27] = [[60,24],[65,21],[70,20],[75,21],[80,24]]
    points[27:31] = [[50,32],[50,39],[50,46],[50,51]]
    points[31:36] = [[42,52],[46,54],[50,55],[54,54],[58,52]]
    points[36:42] = [[20,35],[25,32],[35,32],[40,35],[35,38],[25,38]]
    points[42:48] = [[60,35],[65,32],[75,32],[80,35],[75,38],[65,38]]
    points[48:60] = [[36,68],[40,65],[45,63],[50,64],[55,63],[60,65],[64,68],[60,71],[55,73],[50,74],[45,73],[40,71]]
    points[60:68] = [[41,68],[45,66],[50,67],[55,66],[59,68],[55,70],[50,71],[45,70]]
    return points


def by_id(result):
    return {row["id"]:row for row in result["metrics"]}


def test_expected_geometry_and_unavailable_pupils():
    result = measure(face(),"300w")
    rows = by_id(result)
    assert rows["eye_tilt_0"]["value"] == 0
    assert rows["eye_tilt_1"]["value"] == 0
    assert rows["eye_aspect_0"]["value"] == pytest.approx(20/6,abs=1e-5)
    assert rows["eye_spacing"]["value"] == 1
    assert rows["asymmetry"]["value"] == 0
    assert "midface_pupils" not in rows
    assert any("pupil" in row["term"] for row in result["unavailable"])
    assert all(row["points"] and row["formula"] and row["limitation"] for row in rows.values())


def test_rotation_translation_scale_invariance():
    base = by_id(measure(face(),"300w"))
    angle = .23
    rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    moved = by_id(measure(face()@rotation.T*2.5+[170,120],"300w"))
    for key in base:
        assert moved[key]["value"] == pytest.approx(base[key]["value"],abs=2e-5)


def test_canthal_tilt_sign_with_y_down():
    points = face()
    points[36,1] -= 2
    points[45,1] -= 2
    rows = by_id(measure(points,"300w"))
    assert rows["eye_tilt_0"]["value"] == pytest.approx(math.degrees(math.atan2(2,20)),abs=1e-5)
    assert rows["eye_tilt_1"]["value"] == pytest.approx(math.degrees(math.atan2(2,20)),abs=1e-5)


@pytest.mark.parametrize("name",["300w","wflw"])
def test_flip_is_bijective_and_involutive(name):
    t = make_topology(name)
    assert sorted(t.flip)==list(range(t.count))
    assert [t.flip[t.flip[i]] for i in range(t.count)]==list(range(t.count))


def test_crop_affine_round_trip():
    image = np.zeros((300,400,3),dtype=np.uint8)
    _,transform = crop_image(image,[20,30,100,160])
    points = np.array([[20,30],[100,190],[50,85]],float)
    unit = normalized_points(points,transform)
    assert np.allclose(image_points(unit,transform),points)
    homogeneous = np.c_[points,np.ones(len(points))]
    assert np.allclose(homogeneous@np.array(transform["affine"]).T,unit*95,atol=1e-4)


def test_invalid_and_degenerate_points():
    with pytest.raises(ValueError):
        measure(np.zeros((68,2)),"300w")
    points = face(); points[0,0] = np.nan
    with pytest.raises(ValueError):
        measure(points,"300w")
