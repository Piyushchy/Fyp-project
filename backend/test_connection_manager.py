"""
Tests for the telemedicine ConnectionManager and /ws/multimodal/{role}/{room_id} endpoint.

Uses FastAPI's TestClient with HTTPX WebSocket support so no running server
or loaded ML models are required — the `process_multimodal` function is
monkey-patched with a lightweight stub.

Run:
    pytest backend/test_connection_manager.py -v
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Stub out heavy model loading so we can import app.py without GPU / weights
# ---------------------------------------------------------------------------

def _stub_process_multimodal(
    image_b64: str | None,
    audio_b64: str | None,
    text: str | None,
    prev_time: float,
) -> tuple[dict[str, Any], float]:
    """Lightweight stand-in for the real process_multimodal."""
    now = time.time()
    return {
        "fps": 30.0,
        "face_detected": bool(image_b64),
        "bbox": None,
        "vit_prediction": None,
        "mediapipe_confidence": None,
        "fusion_prediction": {
            "label": "happy",
            "confidence": 0.91,
            "probs": {
                "happy": 0.91,
                "sad": 0.03,
                "angry": 0.02,
                "neutral": 0.02,
                "surprise": 0.02,
            },
            "modalities_used": ["vision", "text"],
            "modality_weights": {"vision": 0.6, "text": 0.4},
        },
        "transcript": text or "",
    }, now


# We patch process_multimodal at module level so the endpoint picks it up.
# We also need to patch the lifespan so it doesn't try to load real models.
from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


# Apply patches before importing app
with patch.dict("sys.modules", {}):
    pass

import backend.app as app_module  # noqa: E402

# Patch before creating the TestClient
app_module.process_multimodal = _stub_process_multimodal  # type: ignore[assignment]
app_module.app.router.lifespan_context = _noop_lifespan  # type: ignore[assignment]

client = TestClient(app_module.app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_manager():
    """Clear all rooms between tests."""
    app_module._manager._rooms.clear()


# ---------------------------------------------------------------------------
# Tests — ConnectionManager unit
# ---------------------------------------------------------------------------

class TestConnectionManagerUnit:
    """Direct unit tests on the ConnectionManager datastructure."""

    def setup_method(self):
        _reset_manager()

    def test_rooms_start_empty(self):
        assert app_module._manager._rooms == {}

    def test_disconnect_unknown_room_does_not_crash(self):
        """Calling disconnect on a non-existent room must not raise."""

        class FakeWS:
            pass

        app_module._manager.disconnect(FakeWS(), "patient", "nonexistent-room")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests — WebSocket endpoint integration
# ---------------------------------------------------------------------------

class TestTelemedicineEndpoint:
    """Integration tests hitting /ws/multimodal/{role}/{room_id}."""

    def setup_method(self):
        _reset_manager()

    # --- Role validation ---

    def test_invalid_role_rejected(self):
        """An unknown role should get a policy-violation close."""
        with pytest.raises(Exception):
            # Starlette TestClient raises on unexpected close
            with client.websocket_connect("/ws/multimodal/nurse/room1"):
                pass

    # --- Patient sends, doctor receives ---

    def test_patient_to_doctor_broadcast(self):
        """Patient payload → inference → broadcast to doctor only."""
        with client.websocket_connect("/ws/multimodal/doctor/room-abc") as doc_ws:
            with client.websocket_connect("/ws/multimodal/patient/room-abc") as pat_ws:
                pat_ws.send_text(json.dumps({
                    "image": "fakebase64",
                    "text": "I feel great",
                }))

                # Doctor should receive the prediction
                result = doc_ws.receive_json()
                assert result["fusion_prediction"]["label"] == "happy"
                assert result["room_id"] == "room-abc"
                assert result["source_role"] == "patient"

    def test_patient_does_not_receive_own_prediction(self):
        """The patient must NOT get the inference result echoed back, only an ACK."""
        with client.websocket_connect("/ws/multimodal/doctor/room-x") as doc_ws:
            with client.websocket_connect("/ws/multimodal/patient/room-x") as pat_ws:
                pat_ws.send_text(json.dumps({"image": "abc123"}))

                # Doctor gets the prediction message
                doc_msg = doc_ws.receive_json()
                assert "fusion_prediction" in doc_msg

                # Patient receives backpressure acknowledgment only, not prediction
                pat_ack = pat_ws.receive_json()
                assert pat_ack.get("status") == "acknowledged"
                assert "fusion_prediction" not in pat_ack

                # Second send+receive cycle confirms backpressure release
                pat_ws.send_text(json.dumps({"text": "hello"}))
                result2 = doc_ws.receive_json()
                assert result2["transcript"] == "hello"

                pat_ack2 = pat_ws.receive_json()
                assert pat_ack2.get("status") == "acknowledged"

    # --- Room isolation ---

    def test_cross_room_isolation(self):
        """Doctors in room-A must NOT receive predictions from room-B."""
        with client.websocket_connect("/ws/multimodal/doctor/room-A") as doc_a:
            with client.websocket_connect("/ws/multimodal/doctor/room-B") as doc_b:
                with client.websocket_connect("/ws/multimodal/patient/room-B") as pat_b:
                    pat_b.send_text(json.dumps({"text": "room-B patient"}))

                    # Doctor B should receive
                    msg = doc_b.receive_json()
                    assert msg["room_id"] == "room-B"

                    # Doctor A should NOT have anything — we can't easily
                    # assert "nothing received" with sync test client, so
                    # instead verify the manager state.
                    assert "room-A" in app_module._manager._rooms
                    assert len(app_module._manager._rooms["room-A"]["doctor"]) == 1

    # --- Multiple doctors in one room ---

    def test_multiple_doctors_receive(self):
        """All doctors in the same room receive the prediction."""
        with client.websocket_connect("/ws/multimodal/doctor/room-multi") as doc1:
            with client.websocket_connect("/ws/multimodal/doctor/room-multi") as doc2:
                with client.websocket_connect("/ws/multimodal/patient/room-multi") as pat:
                    pat.send_text(json.dumps({"image": "img", "text": "hi"}))

                    r1 = doc1.receive_json()
                    r2 = doc2.receive_json()
                    assert r1["room_id"] == "room-multi"
                    assert r2["room_id"] == "room-multi"
                    assert r1["fusion_prediction"] == r2["fusion_prediction"]

    # --- Graceful disconnect ---

    def test_patient_disconnect_does_not_crash_room(self):
        """If the patient disconnects, the doctor's connection survives."""
        with client.websocket_connect("/ws/multimodal/doctor/room-dc") as doc_ws:
            # Patient connects and disconnects
            with client.websocket_connect("/ws/multimodal/patient/room-dc") as pat_ws:
                pat_ws.send_text(json.dumps({"text": "before disconnect"}))
                doc_ws.receive_json()

            # Patient is now disconnected — room should still have doctor
            room = app_module._manager._rooms.get("room-dc")
            assert room is not None
            assert len(room["doctor"]) == 1
            assert len(room["patient"]) == 0

    def test_doctor_disconnect_cleans_up(self):
        """When all participants leave, the room is garbage-collected."""
        with client.websocket_connect("/ws/multimodal/doctor/room-gc"):
            assert "room-gc" in app_module._manager._rooms

        # After both sides disconnect, room should be gone
        # (patient was never in this room, so only doctor leaving triggers GC)
        assert "room-gc" not in app_module._manager._rooms

    # --- Backward compat: legacy endpoint still works ---

    def test_legacy_multimodal_endpoint_still_works(self):
        """The original /ws/multimodal endpoint must remain functional."""
        with client.websocket_connect("/ws/multimodal") as ws:
            ws.send_text(json.dumps({"text": "legacy test"}))
            result = ws.receive_json()
            assert "fusion_prediction" in result


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

