import json
from pathlib import Path

import numpy as np
import pytest

from backend.geometry.reference import ideal_reference
from backend.geometry.rating import CATALOG,RatingRequest,calculate
from backend.geometry.reference_gallery import gallery_signature,gallery_is_current


def test_reference_is_explicitly_constructed_and_every_raw_criterion_is_in_band():
    reference=ideal_reference()
    assert reference["kind"]=="constructed_geometry_not_prediction"
    assert len(reference["points"])==98
    assert reference["rating"]["value"]==10
    assert reference["rating"]["used_criteria"]==5
    points=[[point["x"],point["y"]] for point in reference["points"]]
    recalculated=calculate(RatingRequest(dataset="wflw",points=points,rules=CATALOG))
    assert recalculated["value"]==10
    for part in recalculated["parts"]:
        assert part["low"]<=part["observed"]<=part["high"]
        assert part["deviation"]["direction"]=="within"
        assert part["lost_contribution"]==0
        assert part["score"]==10


def test_target_geometry_preserves_score_under_uniform_resize_and_shift():
    reference=ideal_reference()
    points=np.array([[p["x"],p["y"]] for p in reference["points"]])*.7+[15,60]
    actual=calculate(RatingRequest(dataset="wflw",points=points.tolist(),rules=CATALOG))
    assert actual["value"]==10
    assert [p["observed"] for p in actual["parts"]]==pytest.approx([p["observed"] for p in reference["rating"]["parts"]],abs=1e-5)


def test_published_examples_do_not_claim_perfect_scores_they_did_not_achieve():
    path=Path(__file__).resolve().parents[2]/"reports/reference-values/examples.json"
    if not path.is_file():
        pytest.skip("Optional local research gallery is not installed.")
    gallery=json.loads(path.read_text(encoding="utf-8"))
    assert len(gallery["examples"])==3
    for example in gallery["examples"]:
        all_ten=len(example["parts"])==5 and all(part["deviation"]["direction"]=="within" for part in example["parts"])
        assert example["all_ten"] is all_ten
        if not all_ten:
            assert any(part["score"]<10 for part in example["parts"])
        assert sum(part["contribution"] for part in example["parts"])==pytest.approx(example["value"],abs=.0051)


def test_gallery_invalidates_when_detector_changes_even_if_landmarks_do_not(tmp_path):
    landmark=tmp_path/"landmarks.pt"; landmark.write_bytes(b"own-landmark-weights")
    detector=tmp_path/"detector.pt"; detector.write_bytes(b"own-detector-v1")
    gallery=gallery_signature(landmark,detector)
    assert gallery_is_current(gallery,landmark,detector)
    detector.write_bytes(b"own-detector-v2")
    assert not gallery_is_current(gallery,landmark,detector)


def test_gallery_invalidates_stale_rules_pipeline_or_manual_gt_crops(tmp_path):
    landmark=tmp_path/"landmarks.pt"; landmark.write_bytes(b"own-weights")
    detector=tmp_path/"detector.pt"; detector.write_bytes(b"own-detector")
    current=gallery_signature(landmark,detector)
    for key in ["checkpoint_sha256","detector_checkpoint_sha256","detector_pipeline_version","rating_rules_sha256","scan_mode"]:
        assert not gallery_is_current({**current,key:"outdated"},landmark,detector)
    assert not gallery_is_current({},landmark,detector)
