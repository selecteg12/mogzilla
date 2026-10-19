import numpy as np
import pytest

from backend.ml.face_detector import encode_box, decode_box, inside_fraction, proposals
from backend.ml import face_detector


def test_regressed_box_round_trip_and_source_pixel_coordinates():
    face = [120., 90., 80., 160.]
    window = [80., 60., 200., 200.]
    encoded = encode_box(face, window)
    assert encoded[:2] == pytest.approx([-.1, .05])
    assert np.exp(encoded[2:]) == pytest.approx([.4, .8])
    assert decode_box(encoded, window) == pytest.approx(face, abs=1e-5)
    # Mirroring a proposal must mirror its regressed face, keeping the size.
    encoded[0] *= -1
    assert decode_box(encoded, window) == pytest.approx([160., 90., 80., 160.], abs=1e-5)


def test_partial_face_coverage_is_not_confused_with_iou():
    face = [50., 50., 100., 160.]
    forehead_and_eyes = [45., 30., 115., 100.]
    full_context = [20., 20., 200., 240.]
    assert inside_fraction(face, forehead_and_eyes) == .5
    assert inside_fraction(face, full_context) == 1


def test_pyramid_includes_context_for_large_portraits_and_is_finite():
    windows = proposals(np.zeros((920, 736, 3), dtype=np.uint8))
    assert max(window[2] for window in windows) > 736
    assert np.isfinite(windows).all()
    assert all(window[2] == window[3] and window[2] > 0 for window in windows)


def test_failed_first_proposal_cannot_suppress_verified_whole_face(monkeypatch):
    windows=[[0.,0.,200.,200.]]*4
    whole=[45.,45.,100.,135.]
    partial=[55.,45.,90.,90.]
    calls=0
    def predictions(model,rgb,crops):
        nonlocal calls
        calls+=1
        if calls==1:
            return [1.,.99,.98,.97],[partial,whole,whole,whole]
        if calls==2:
            return [.1,.99,.99,.99],[partial,whole,whole,whole]
        return [.99],[whole]
    monkeypatch.setattr(face_detector,"proposals",lambda rgb: windows)
    monkeypatch.setattr(face_detector,"predict_windows",predictions)
    result=face_detector.find_faces(None,np.zeros((220,220,3),dtype=np.uint8))
    assert result["status"]=="found"
    assert result["candidates"][0]["box"]==whole
    assert result["candidates"][0]["supporting_windows"]==3
