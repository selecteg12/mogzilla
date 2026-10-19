import numpy as np
import pytest
from pydantic import ValidationError

from backend.geometry.rating import RatingRequest,RatingRule,calculate,criterion_score
from backend.tests.test_geometry import face
from backend.ml.regions import padded_box,patch
from backend.preprocessing.images import crop_image


def rule(**kwargs):
    return RatingRule(id="spacing",low=.95,high=1.05,falloff=.6,weight=1,**kwargs)


def request(rules,reviewed=True):
    return RatingRequest(dataset="300w",points=face().tolist(),rules=rules,landmarks_reviewed=reviewed)


def test_score_band_falloff_and_limits():
    r=rule()
    assert criterion_score(1,r)==10
    assert criterion_score(.65,r)==pytest.approx(5)
    assert criterion_score(1.35,r)==pytest.approx(5)
    assert criterion_score(2,r)==0


def test_rating_is_immediate_and_review_only_changes_label():
    provisional=calculate(request([rule()],False))
    reviewed=calculate(request([rule()],True))
    assert provisional["value"]==reviewed["value"]==10
    assert provisional["provisional"] is True
    assert reviewed["provisional"] is False


def test_missing_pupils_are_not_invented():
    pupil=RatingRule(id="midface_pupils",low=.93,high=1.01,falloff=.45,weight=1)
    rated=calculate(request([rule(),pupil]))
    assert rated["value"]==10
    assert rated["used_criteria"]==1
    assert rated["unavailable"][0]["id"]=="midface_pupils"


def test_weighted_contributions_and_sensitivity():
    # Fixture has spacing 1 (score 10), tilt 0 (score 5 for this chosen band).
    tilt=RatingRule(id="tilt_mean",low=5,high=8,falloff=10,weight=3)
    errors={"eye_spacing":{"mean_absolute_error":.1},"eye_tilt_0":{"mean_absolute_error":2},"eye_tilt_1":{"mean_absolute_error":2}}
    rated=calculate(request([rule(),tilt]),errors)
    assert rated["value"]==6.25
    assert sum(p["contribution"] for p in rated["parts"])==pytest.approx(6.25)
    assert sum(p["lost_contribution"] for p in rated["parts"])==pytest.approx(3.75)
    assert rated["sensitivity"]["low"]<6.25<rated["sensitivity"]["high"]
    assert "не доверительный" in rated["sensitivity"]["interpretation"]
    tilt.weight=0
    assert calculate(request([rule(),tilt]))["value"]==10
    assert calculate(request([tilt]))["value"] is None


def test_invalid_geometry_and_preferences_fail():
    with pytest.raises(ValidationError):
        RatingRule(id="spacing",low=2,high=1,falloff=1,weight=1)
    with pytest.raises(ValidationError):
        RatingRule(id="spacing",low=1,high=2,falloff=0,weight=1)
    with pytest.raises(ValidationError):
        request([rule(),rule()])
    with pytest.raises(ValidationError):
        RatingRequest(dataset="wflw",points=face().tolist(),rules=[rule()])
    degenerate=request([rule()]); degenerate.points=np.zeros((68,2)).tolist()
    with pytest.raises(ValueError):
        calculate(degenerate)


def test_search_patch_matches_landmark_crop():
    # The detector's square window and landmark model must see identical pixels.
    rgb=np.random.default_rng(1).integers(0,256,(200,250,3),dtype=np.uint8)
    box=[31,40,80,100]
    crop,_=crop_image(rgb,box)
    assert np.array_equal(crop,patch(rgb,padded_box(box)))


def test_photo_annotations_follow_original_points_and_transform():
    tilt=RatingRule(id="tilt_mean",low=5,high=8,falloff=10,weight=1)
    mouth=RatingRule(id="mouth_nose",low=1.3,high=1.6,falloff=1,weight=1)
    payload=request([rule(),tilt,mouth])
    base=calculate(payload)
    angle=.25
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    payload.points=(face()@rotation.T*2+[40,80]).tolist()
    moved=calculate(payload)
    assert moved["value"]==base["value"]
    for original, transformed in zip(base["parts"],moved["parts"]):
        assert original["description"]
        assert original["deviation"]["direction"]==transformed["deviation"]["direction"]
        assert len(original["annotations"])>=2
        for line, translated in zip(original["annotations"],transformed["annotations"]):
            assert np.allclose(np.array(line["a"])@rotation.T*2+[40,80],translated["a"])
            assert np.allclose(np.array(line["b"])@rotation.T*2+[40,80],translated["b"])
    mouth_part=next(p for p in base["parts"] if p["id"]=="mouth_nose")
    lengths=[np.linalg.norm(np.array(line["a"])-line["b"]) for line in mouth_part["annotations"]]
    assert lengths[0]/lengths[1]==pytest.approx(mouth_part["observed"],abs=1e-5)
