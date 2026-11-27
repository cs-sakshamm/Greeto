"""
Phase 3 Milestone 2 recognition checks.

Pure numpy and in-memory fakes: no camera, no InsightFace model, no GUI.
Run with:

    python tests/test_recognition.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

import numpy as np

from app.vision.face_database import FaceDatabase, sanitize_embeddings
from app.vision.face_recognizer import FaceRecognizer
from app.vision.models import Face
from app.vision.recognition import (
    RecognitionConfig,
    calibrate_threshold,
)
from app.vision.similarity import cosine_similarity, normalize_embedding
from app.vision.visitor_tracker import VisitorTracker


FAILURES = []


def check(name, fn):
    """Run one check and record the outcome."""

    started = time.perf_counter()

    try:

        fn()

        elapsed = time.perf_counter() - started

        print("[PASS] %-58s (%.2fs)" % (name, elapsed))

    except Exception as error:  # noqa: BLE001 - test harness reports everything

        print("[FAIL] %-58s -> %s: %s" % (name, type(error).__name__, error))

        FAILURES.append(name)


def expect_value_error(fn, label):
    """Assert that a call raises ValueError."""

    try:

        fn()

    except ValueError:

        return

    raise AssertionError("%s did not raise ValueError" % label)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

SIZE = 512


def unit(index, size=SIZE):
    """One-hot unit vector."""

    vector = np.zeros(size, dtype=np.float32)

    vector[index] = 1.0

    return vector


def face(embedding, confidence=0.90, width=200, height=200):
    return Face(
        x=100,
        y=100,
        width=width,
        height=height,
        embedding=embedding,
        confidence=confidence,
    )


class FakeDatabase:
    def __init__(self, mapping):

        self.mapping = mapping

    def get_database(self):

        return self.mapping


class FakeConfig:
    def __init__(self, mapping=None):

        self.mapping = mapping

    def get_value(self, file_name, *keys, default=None):

        if file_name == "vision" and keys == ("recognition",):

            return self.mapping

        return default


class FakeCache:
    def __init__(self, data=None, fail=False, exists=True):

        self.data = data

        self.fail = fail

        self._exists = exists

        self.saved = []

    def exists(self):

        return self._exists

    def load(self):

        if self.fail:

            raise RuntimeError("corrupt pickle")

        return self.data

    def save(self, database):

        self.saved.append(database)


class FakeModel:
    def generate(self, image):

        return None


class FakeLogger:
    def __getattr__(self, name):

        return lambda *args, **kwargs: None


class FakeCamera:
    failure_threshold = 3

    reconnect_interval = 0.01

    def status(self):

        return {"state": "ready"}

    def release(self):

        return None

    def is_open(self):

        return False


class FakeBus:
    """Minimal EventBus stand-in that records event payloads."""

    def __init__(self, fail: bool = False):

        self.events = []

        self.fail = fail

    def subscribe(self, event_name, callback):

        pass

    def publish(self, event_name, **payload):

        if self.fail:

            raise RuntimeError("event bus unavailable")

        self.events.append((event_name, payload))


class FakeContainer:
    def __init__(self):

        self.config = FakeConfig(None)

        self.logger = FakeLogger()

        self.event_bus = FakeBus()


class FakeVisitorDatabase:
    def __init__(self, crash: bool = False):

        self.registered = []

        self.crash = crash

    def register(self, name):

        if self.crash:

            raise RuntimeError("visitor database unavailable")

        self.registered.append(name)

    def stats(self):

        return {
            "exists": True,
            "visitors": len(set(self.registered)),
            "visits": len(self.registered),
        }


# ----------------------------------------------------------------------
# A. Embedding validation and similarity
# ----------------------------------------------------------------------

def test_normalize_embedding():

    raw = np.array([3.0, 4.0], dtype=np.float32)

    vector = normalize_embedding(raw)

    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5

    assert vector.dtype == np.float32

    scaled = normalize_embedding(raw * 7.0)

    assert np.allclose(vector, scaled, atol=1e-6)

    unnormalised = normalize_embedding(raw, normalize=False)

    assert np.allclose(unnormalised, raw)

    # Explicit size is enforced.
    assert normalize_embedding(unit(0), expected_size=SIZE).size == SIZE

    expect_value_error(
        lambda: normalize_embedding(unit(0), expected_size=128),
        "wrong embedding size",
    )

    expect_value_error(lambda: normalize_embedding([]), "empty embedding")

    expect_value_error(
        lambda: normalize_embedding(np.zeros(SIZE)),
        "zero embedding",
    )

    expect_value_error(
        lambda: normalize_embedding(np.full(SIZE, np.nan)),
        "nan embedding",
    )

    expect_value_error(
        lambda: normalize_embedding(np.ones((2, SIZE))),
        "two-dimensional embedding",
    )

    expect_value_error(
        lambda: normalize_embedding(["not", "numbers"]),
        "non-numeric embedding",
    )

    print("       normalised vectors are unit length and scale-invariant")


def test_cosine_similarity():

    first = unit(0)

    second = unit(1)

    assert abs(cosine_similarity(first, first) - 1.0) < 1e-6

    assert abs(cosine_similarity(first, second) - 0.0) < 1e-6

    assert abs(cosine_similarity(first, -first) + 1.0) < 1e-6

    # Scaling must not change the result.
    assert abs(cosine_similarity(first * 12.0, first) - 1.0) < 1e-6

    assert -1.0 <= cosine_similarity(np.ones(4), np.arange(4)) <= 1.0

    expect_value_error(
        lambda: cosine_similarity(first, unit(1, size=8)),
        "mismatched sizes",
    )

    expect_value_error(
        lambda: cosine_similarity(np.zeros(4), np.ones(4)),
        "zero vector",
    )

    expect_value_error(
        lambda: cosine_similarity(np.full(4, np.inf), np.ones(4)),
        "infinite vector",
    )

    print("       cosine scores stay finite and bounded")


# ----------------------------------------------------------------------
# B. Configuration
# ----------------------------------------------------------------------

def test_config_defaults_and_overrides():

    defaults = RecognitionConfig.from_mapping({})

    assert defaults.threshold == 0.55

    assert defaults.min_detection_score == 0.55

    assert defaults.min_face_size == 60

    assert defaults.expected_embedding_size == 512

    assert defaults.normalize_embeddings is True

    assert defaults.warnings == ()

    configured = RecognitionConfig.from_mapping(
        {
            "threshold": 0.72,
            "min_detection_score": 0.66,
            "min_face_size": 90,
            "expected_embedding_size": 128,
            "min_embedding_norm": 0.25,
            "normalize_embeddings": "false",
        }
    )

    assert configured.threshold == 0.72

    assert configured.min_detection_score == 0.66

    assert configured.min_face_size == 90

    assert configured.expected_embedding_size == 128

    assert configured.min_embedding_norm == 0.25

    assert configured.normalize_embeddings is False

    # Strings that look numeric are accepted for numbers.
    assert RecognitionConfig.from_mapping({"threshold": "0.61"}).threshold == 0.61

    # Bad values fall back, with a warning, instead of crashing startup.
    invalid = RecognitionConfig.from_mapping(
        {
            "threshold": 4.0,
            "min_detection_score": "high",
            "min_face_size": -2,
            "expected_embedding_size": 0,
            "min_embedding_norm": 0,
            "normalize_embeddings": "sometimes",
        }
    )

    assert invalid.threshold == 0.55

    assert invalid.min_detection_score == 0.55

    assert invalid.min_face_size == 60

    assert invalid.expected_embedding_size == 512

    assert invalid.min_embedding_norm == 0.5

    assert invalid.normalize_embeddings is True

    assert len(invalid.warnings) >= 6

    # ConfigManager lookups go through vision.recognition.
    from_config = RecognitionConfig.from_config(
        FakeConfig({"threshold": 0.68})
    )

    assert from_config.threshold == 0.68

    assert RecognitionConfig.from_config(FakeConfig("not a mapping")).warnings

    assert RecognitionConfig.from_config(None).threshold == 0.55

    assert RecognitionConfig().with_threshold(0.8).threshold == 0.8

    expect_value_error(
        lambda: RecognitionConfig().with_threshold(float("nan")),
        "nan threshold",
    )

    print("       invalid config values fall back with warnings")


# ----------------------------------------------------------------------
# C. Recognition results and quality gates
# ----------------------------------------------------------------------

def test_recognized_and_unknown():

    alice = unit(0)

    database = FakeDatabase({"Alice": [alice * 3.0], "Bob": [unit(1)]})

    recognizer = FaceRecognizer(database, threshold=0.55)

    assert recognizer.refresh() == (2, 2, 0)

    known = face(alice)

    result = recognizer.recognize_result(known)

    assert result.recognized is True

    assert result.status == "recognized"

    assert result.name == "Alice"

    assert abs(result.score - 1.0) < 1e-5

    recognizer.recognize(known)

    assert known.name == "Alice"

    assert known.recognized is True

    assert known.recognition_status == "recognized"

    # Detection confidence is preserved; similarity is separate.
    assert known.confidence == 0.90

    assert known.recognition_score is not None

    stranger = recognizer.recognize(face(unit(2)))

    assert stranger.recognized is False

    assert stranger.name == "Unknown"

    assert stranger.recognition_status == "unknown"

    assert stranger.recognition.candidate == "Alice"

    assert stranger.recognition.score is not None

    assert 0.0 <= stranger.recognition.score < 0.55

    print("       recognized/unknown are distinct structured outcomes")


def test_quality_gates():

    database = FakeDatabase({"Alice": [unit(0)]})

    recognizer = FaceRecognizer(database, threshold=0.55)

    low_detection = recognizer.recognize(face(unit(0), confidence=0.30))

    assert low_detection.recognized is False

    assert low_detection.recognition_status == "low_quality"

    assert "detection score" in low_detection.recognition.reason

    assert low_detection.recognition.score is None

    tiny = recognizer.recognize(face(unit(0), width=20, height=20))

    assert tiny.recognition_status == "low_quality"

    assert "minimum side" in tiny.recognition.reason

    zero_box = recognizer.recognize(face(unit(0), width=0, height=200))

    assert zero_box.recognition_status == "low_quality"

    assert "bounding box" in zero_box.recognition.reason

    nan_score = recognizer.recognize(face(unit(0), confidence=float("nan")))

    assert nan_score.recognition_status == "low_quality"

    print("       low-quality faces never reach similarity matching")


def test_invalid_embeddings():

    database = FakeDatabase({"Alice": [unit(0)]})

    recognizer = FaceRecognizer(database, threshold=0.55)

    for embedding, label in (
        (np.zeros(SIZE), "zero"),
        (np.full(SIZE, np.nan), "nan"),
        (np.ones(128), "wrong size"),
        (np.ones((1, SIZE)), "wrong shape"),
    ):

        result = recognizer.recognize_result(face(embedding))

        assert result.recognized is False

        assert result.status == "invalid_embedding", label

        assert result.score is None

    print("       malformed embeddings are reported, not compared")


def test_empty_and_broken_database():

    empty = FaceRecognizer(FakeDatabase({}), threshold=0.55)

    result = empty.recognize_result(face(unit(0)))

    assert result.status == "no_known_faces"

    assert result.recognized is False

    class BrokenDatabase:
        def get_database(self):

            return None

    broken = FaceRecognizer(BrokenDatabase(), threshold=0.55)

    result = broken.recognize_result(face(unit(0)))

    assert result.status == "no_known_faces"

    assert "unavailable" in result.reason

    print("       empty/broken databases degrade to unknown")


def test_batch_never_aborts():

    database = FakeDatabase({"Alice": [unit(0)]})

    recognizer = FaceRecognizer(database, threshold=0.55)

    faces = [
        face(unit(0)),
        face(np.zeros(SIZE)),
        face(unit(0), confidence=0.1),
        face(unit(0)),
    ]

    results = recognizer.recognize_all(faces)

    assert len(results) == 4

    assert results[0].recognized is True

    assert results[1].recognition_status == "invalid_embedding"

    assert results[2].recognition_status == "low_quality"

    assert results[3].recognized is True

    print("       one bad face does not abort the batch")


def test_threshold_controls_decision():

    query = unit(0) * 0.8 + unit(1) * 0.6  # cosine 0.8 with Alice

    database = FakeDatabase({"Alice": [unit(0)]})

    lenient = FaceRecognizer(database, threshold=0.70)

    strict = FaceRecognizer(database, threshold=0.90)

    assert lenient.recognize_result(face(query)).recognized is True

    strict_result = strict.recognize_result(face(query))

    assert strict_result.recognized is False

    assert strict_result.status == "unknown"

    assert abs(strict_result.score - 0.8) < 1e-5

    print("       threshold is applied at the decision boundary")


def test_stats():

    database = FakeDatabase({"Alice": [unit(0)], "Bob": [np.zeros(SIZE)]})

    recognizer = FaceRecognizer(database, threshold=0.66)

    identities, embeddings, rejected = recognizer.refresh()

    assert identities == 1

    assert embeddings == 1

    assert rejected == 1

    recognizer.recognize(face(unit(0)))

    stats = recognizer.stats()

    assert stats["threshold"] == 0.66

    assert stats["known_identities"] == 1

    assert stats["known_embeddings"] == 1

    assert stats["last_status"] == "recognized"

    assert stats["last_score"] is not None

    print("       stats expose threshold, database size and last outcome")


# ----------------------------------------------------------------------
# D. Unique visitor IDs
# ----------------------------------------------------------------------

def tracker_face(x, y=100):
    return Face(
        x=x - 50,
        y=y - 50,
        width=100,
        height=100,
        embedding=unit(0),
        confidence=0.9,
    )


def test_two_overlapping_faces_get_distinct_ids():

    tracker = VisitorTracker()

    faces = [tracker_face(100), tracker_face(100)]

    tracker.update(faces)

    ids = [item.visitor_id for item in faces]

    assert len(set(ids)) == 2, ids

    assert ids == [1, 2], ids

    # The next frame keeps both tracks stable.
    faces = [tracker_face(100), tracker_face(100)]

    tracker.update(faces)

    assert [item.visitor_id for item in faces] == [1, 2]

    print("       overlapping faces receive different IDs: %s" % ids)


def test_many_faces_same_centroid():

    tracker = VisitorTracker()

    faces = [tracker_face(200) for _ in range(50)]

    tracker.update(faces)

    ids = [item.visitor_id for item in faces]

    assert len(set(ids)) == 50

    assert sorted(ids) == list(range(1, 51))

    print("       50 faces -> 50 unique IDs")


def test_expired_ids_are_not_reused():

    tracker = VisitorTracker(max_missing_frames=1)

    first = tracker_face(100)

    tracker.update([first])

    assert first.visitor_id == 1

    tracker.update([])

    tracker.update([])

    assert tracker.active_track_count() == 0

    second = tracker_face(100)

    tracker.update([second])

    assert second.visitor_id == 2, "expired ID was reused"

    tracker.clear()

    third = tracker_face(100)

    tracker.update([third])

    assert third.visitor_id == 1, "clear() must start a new session"

    print("       expired IDs are never reused within a session")


def test_tracks_survive_movement():

    tracker = VisitorTracker(max_distance=100.0)

    face = tracker_face(100)

    tracker.update([face])

    for x in (110, 125, 140, 155):

        moved = tracker_face(x)

        tracker.update([moved])

        assert moved.visitor_id == 1

    assert tracker.active_track_count() == 1

    assert tracker.has_active_tracks() is True

    print("       centroid movement keeps the same visitor ID")


# ----------------------------------------------------------------------
# E. Threshold calibration
# ----------------------------------------------------------------------

def test_calibration_clear_separation():

    result = calibrate_threshold(
        genuine_scores=[0.80, 0.85, 0.90, 0.95],
        impostor_scores=[0.10, 0.20, 0.30, 0.40],
    )

    assert result is not None

    assert 0.40 < result.threshold < 0.80

    assert result.balanced_accuracy == 1.0

    assert result.false_accept_rate == 0.0

    assert result.false_reject_rate == 0.0

    assert result.genuine_count == 4

    assert result.impostor_count == 4

    # The chosen threshold sits in the widest part of the gap.
    assert abs(result.threshold - 0.60) < 0.01

    print("       calibrated threshold %.2f (BA %.2f)"
          % (result.threshold, result.balanced_accuracy))


def test_calibration_overlap():

    result = calibrate_threshold(
        genuine_scores=[0.60, 0.70, 0.80],
        impostor_scores=[0.55, 0.65, 0.75],
    )

    assert result is not None

    assert 0.0 <= result.balanced_accuracy <= 1.0

    assert 0.0 <= result.false_accept_rate <= 1.0

    assert 0.0 <= result.false_reject_rate <= 1.0

    print("       overlapping distributions yield a compromise threshold")


def test_calibration_requires_impostors():

    assert calibrate_threshold([0.9, 0.95], []) is None

    assert calibrate_threshold([], [0.1, 0.2]) is None

    assert calibrate_threshold([], []) is None

    assert calibrate_threshold([float("nan")], [0.1]) is None

    expect_value_error(
        lambda: calibrate_threshold([0.9], [0.1], low=1.0, high=0.0),
        "invalid calibration range",
    )

    # Non-finite samples are ignored rather than poisoning the result.
    result = calibrate_threshold(
        [0.8, float("nan"), 0.9],
        [0.2, float("inf"), 0.3],
    )

    assert result is not None

    assert result.genuine_count == 2

    assert result.impostor_count == 2

    print("       calibration refuses to guess without impostors")


# ----------------------------------------------------------------------
# F. Cached embedding sanitation
# ----------------------------------------------------------------------

def test_sanitize_embeddings():

    clean, rejected = sanitize_embeddings(
        {
            "Alice": [
                np.ones(8),
                np.zeros(8),
                np.full(8, np.nan),
                np.ones((1, 8)),
            ],
            "Bob": [],
            "": [np.ones(8)],
            7: [np.ones(8)],
        }
    )

    assert list(clean) == ["Alice"]

    assert len(clean["Alice"]) == 1

    assert rejected >= 5

    # A single numpy array is accepted as a one-embedding list.
    clean, rejected = sanitize_embeddings({"Carol": np.ones(8)})

    assert len(clean["Carol"]) == 1

    assert rejected == 0

    clean, rejected = sanitize_embeddings("not a mapping")

    assert clean == {}

    assert rejected == 1

    print("       malformed cached embeddings are dropped")


def test_database_cache_fallbacks():

    with tempfile.TemporaryDirectory() as folder:

        # Unreadable cache: rebuild from an empty folder, no crash.
        database = FaceDatabase(
            FakeModel(),
            database_path=folder,
            cache=FakeCache(fail=True),
        )

        database.load()

        assert database.person_count() == 0

        # Valid cache: used directly and not rebuilt.
        cache = FakeCache(data={"Alice": [np.ones(8)]})

        database = FaceDatabase(
            FakeModel(),
            database_path=folder,
            cache=cache,
        )

        database.load()

        assert database.person_count() == 1

        assert cache.saved == []

        # Partly invalid cache: sanitised and rewritten.
        cache = FakeCache(
            data={"Alice": [np.ones(8), np.zeros(8)]}
        )

        database = FaceDatabase(
            FakeModel(),
            database_path=folder,
            cache=cache,
        )

        database.load()

        assert database.embedding_count() == 1

        assert len(cache.saved) == 1

        # Empty cache: falls through to the (empty) face folder.
        cache = FakeCache(data={"Alice": []})

        database = FaceDatabase(
            FakeModel(),
            database_path=folder,
            cache=cache,
        )

        database.load()

        assert database.person_count() == 0

    print("       corrupt and empty caches fall back safely")


# ----------------------------------------------------------------------
# G. Pipeline integration: structured events and unique IDs
# ----------------------------------------------------------------------

def test_structured_visitor_events():

    from app.vision.vision_service import VisionService

    container = FakeContainer()

    vision = VisionService(
        container,
        camera=FakeCamera(),
        embedding_model=FakeModel(),
        database=FakeDatabase({"Alice": [unit(0)]}),
        visitor_database=FakeVisitorDatabase(),
    )

    vision.recognizer.refresh()

    faces = vision.recognizer.recognize_all(
        [face(unit(0)), face(unit(0))]
    )

    vision.tracker.update(faces)

    assert [item.visitor_id for item in faces] == [1, 2]

    vision._publish_events(faces)

    detected = [
        payload
        for event_name, payload in container.event_bus.events
        if event_name == "visitor.detected"
    ]

    assert len(detected) == 2, detected

    assert [payload["visitor_id"] for payload in detected] == [1, 2]

    for payload in detected:

        assert payload["name"] == "Alice"

        assert payload["status"] == "recognized"

        assert abs(payload["score"] - 1.0) < 1e-5

        assert payload["face"].recognition is not None

    # The same visitors must not be announced again on the next frame.
    vision._publish_events(faces)

    detected_again = [
        payload
        for event_name, payload in container.event_bus.events
        if event_name == "visitor.detected"
    ]

    assert len(detected_again) == 2

    assert vision.visitor_database.registered == ["Alice", "Alice"]

    print("       events carry visitor_id, score and recognition status")


def test_health_includes_recognition():

    from app.vision.vision_service import VisionService

    container = FakeContainer()

    vision = VisionService(
        container,
        camera=FakeCamera(),
        embedding_model=FakeModel(),
        database=FakeDatabase({"Alice": [unit(0)]}),
        visitor_database=FakeVisitorDatabase(),
    )

    vision._running = True
    vision.visitor_database.register("Alice")

    health = vision.health()

    assert health["running"] is True

    assert health["camera"]["state"] == "ready"

    assert health["recognition"]["threshold"] == 0.55

    assert health["recognition"]["known_identities"] == 1

    assert health["visitors"]["visits"] == 1

    print("       health() reports camera, grabber, recognition and visitors")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():

    print("Greeto recognition checks")

    check("A1. embedding validation and normalisation", test_normalize_embedding)

    check("A2. cosine similarity guards", test_cosine_similarity)

    check("B1. config defaults, overrides and fallbacks",
          test_config_defaults_and_overrides)

    check("C1. recognized vs unknown results", test_recognized_and_unknown)

    check("C2. detection/size quality gates", test_quality_gates)

    check("C3. invalid embeddings", test_invalid_embeddings)

    check("C4. empty and broken databases", test_empty_and_broken_database)

    check("C5. batch never aborts", test_batch_never_aborts)

    check("C6. configurable threshold decision", test_threshold_controls_decision)

    check("C7. recognizer stats", test_stats)

    check("D1. overlapping faces get unique IDs",
          test_two_overlapping_faces_get_distinct_ids)

    check("D2. many faces at one centroid", test_many_faces_same_centroid)

    check("D3. expired IDs are not reused", test_expired_ids_are_not_reused)

    check("D4. IDs survive movement", test_tracks_survive_movement)

    check("E1. calibration with clear separation",
          test_calibration_clear_separation)

    check("E2. calibration with overlap", test_calibration_overlap)

    check("E3. calibration needs impostors", test_calibration_requires_impostors)

    check("F1. cached embedding sanitation", test_sanitize_embeddings)

    check("F2. database cache fallbacks", test_database_cache_fallbacks)

    check("G1. structured visitor events", test_structured_visitor_events)

    check("G2. health includes recognition", test_health_includes_recognition)

    print("-" * 72)

    if FAILURES:

        print("RESULT: %d FAILED -> %s" % (len(FAILURES), FAILURES))

        return 1

    print("RESULT: ALL CHECKS PASSED")

    return 0


if __name__ == "__main__":

    sys.exit(main())
