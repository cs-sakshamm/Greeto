"""
Phase 3 Milestone 3 enrolment and admin checks.

Uses small synthetic images and a fake detector, so no InsightFace model,
camera or real face database is touched. The CLI checks exercise the
real argument parser and PIN gate, but never load the model.

    python tests/test_enrollment.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

from app.core.admin_auth import (  # noqa: E402
    AdminAuthError,
    hash_pin,
    load_pin_hash,
    save_pin_hash,
    validate_pin,
    verify_pin,
)
from app.vision.enrollment import (  # noqa: E402
    EnrollmentError,
    EnrollmentService,
    collect_source_images,
    list_identities,
    list_identity_images,
    remove_identity,
    validate_identity_name,
)
from app.vision.face_cache import FaceCache  # noqa: E402
from app.vision.models import Face  # noqa: E402
from app.vision.recognition import RecognitionConfig  # noqa: E402


FAILURES = []

SIZE = 4


def check(name, fn):
    """Run one check and record the outcome."""

    started = time.perf_counter()

    try:

        fn()

        elapsed = time.perf_counter() - started

        print("[PASS] %-62s (%.2fs)" % (name, elapsed))

    except Exception as error:  # noqa: BLE001 - harness reports everything

        print("[FAIL] %-62s -> %s: %s" % (name, type(error).__name__, error))

        FAILURES.append(name)


# ----------------------------------------------------------------------
# Synthetic images and detector
# ----------------------------------------------------------------------

def write_image(path, mode=1):
    """Write a synthetic PNG whose top-left pixel encodes its scenario."""

    image = np.full((240, 240, 3), 40, dtype=np.uint8)

    image[0, 0] = (mode, mode, mode)

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    assert cv2.imwrite(str(path), image)

    return path


def make_face(confidence=0.95, side=120):
    """A valid unit-norm fake face."""

    return Face(
        x=20,
        y=20,
        width=side,
        height=side,
        embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        confidence=confidence,
    )


class FakeModel:
    """Interprets the encoded top-left pixel instead of using InsightFace."""

    def detect(self, image):

        mode = int(image[0, 0, 0])

        if mode == 0:

            return []

        if mode == 2:

            return [make_face(), make_face()]

        if mode == 3:

            return [make_face(confidence=0.20)]

        if mode == 4:

            return [make_face(side=30)]

        return [make_face()]

    def generate(self, image):

        faces = self.detect(image)

        return faces[0].embedding if faces else None


def config():
    return RecognitionConfig(
        threshold=0.55,
        min_detection_score=0.55,
        min_face_size=60,
        expected_embedding_size=SIZE,
        min_embedding_norm=0.5,
        normalize_embeddings=True,
    )


def service(tmp, cache=None):
    return EnrollmentService(
        FakeModel(),
        faces_directory=Path(tmp) / "faces",
        cache=cache or FaceCache(str(Path(tmp) / "cache.pkl")),
        config=config(),
    )


# ----------------------------------------------------------------------
# A. Names
# ----------------------------------------------------------------------

def test_identity_names():

    assert validate_identity_name(" Asha   Rao ") == "Asha Rao"
    assert validate_identity_name("O'Brien") == "O'Brien"
    assert validate_identity_name("Dr. Rao") == "Dr. Rao"

    invalid = [
        "",
        "   ",
        None,
        7,
        "..",
        ".hidden",
        "a/b",
        "a\\b",
        "a:b",
        "a?b",
        "x" * 65,
        "line\nbreak",
        "CON",
        "lpt1",
    ]

    for name in invalid:

        try:

            validate_identity_name(name)

        except EnrollmentError:

            pass

        else:

            raise AssertionError("accepted invalid identity %r" % (name,))


# ----------------------------------------------------------------------
# B. Enrolment writes
# ----------------------------------------------------------------------

def test_valid_enrolment_writes_images_and_cache():

    with tempfile.TemporaryDirectory() as tmp:

        cache = FaceCache(str(Path(tmp) / "cache.pkl"))

        enrollment = service(tmp, cache)

        one = write_image(Path(tmp) / "photos" / "1.png", 1)
        two = write_image(Path(tmp) / "photos" / "2.png", 1)

        result = enrollment.enroll("Alice", [one, two])

        assert result.name == "Alice"
        assert result.added == 2
        assert result.total == 2
        assert result.replaced is False
        assert result.cache_rebuilt is True, result.warning

        files = list_identity_images(Path(tmp) / "faces" / "Alice")

        assert [path.name for path in files] == ["001.jpg", "002.jpg"]

        assert cache.exists()

        cached = cache.load()

        assert "Alice" in cached
        assert len(cached["Alice"]) == 2

        print("       2 images written, cache rebuilt with 2 embeddings")


def test_duplicate_source_paths_are_collapsed():

    with tempfile.TemporaryDirectory() as tmp:

        enrollment = service(tmp)

        photo = write_image(Path(tmp) / "one.png")

        result = enrollment.enroll("Alice", [photo, photo])

        assert result.added == 1

        assert len(list_identity_images(Path(tmp) / "faces" / "Alice")) == 1


def test_add_and_replace():

    with tempfile.TemporaryDirectory() as tmp:

        enrollment = service(tmp)

        first = write_image(Path(tmp) / "first.png", 1)
        second = write_image(Path(tmp) / "second.png", 1)
        replacement = write_image(Path(tmp) / "replacement.png", 5)

        enrollment.enroll("Alice", [first])

        added = enrollment.enroll("Alice", [second])

        assert added.total == 2
        assert added.replaced is False

        files = list_identity_images(Path(tmp) / "faces" / "Alice")

        assert [path.name for path in files] == ["001.jpg", "002.jpg"]

        replaced = enrollment.enroll(
            "Alice",
            [replacement],
            replace=True,
        )

        assert replaced.total == 1
        assert replaced.replaced is True

        files = list_identity_images(Path(tmp) / "faces" / "Alice")

        assert [path.name for path in files] == ["001.jpg"]


# ----------------------------------------------------------------------
# C. Rejection is all-or-nothing
# ----------------------------------------------------------------------

def test_bad_batch_writes_nothing():

    scenarios = [
        (0, "no face"),
        (2, "two faces"),
        (3, "low detection score"),
        (4, "tiny face"),
    ]

    for mode, label in scenarios:

        with tempfile.TemporaryDirectory() as tmp:

            enrollment = service(tmp)

            good = write_image(Path(tmp) / "good.png", 1)
            bad = write_image(Path(tmp) / "bad.png", mode)

            try:

                enrollment.enroll("Alice", [good, bad])

            except EnrollmentError:

                pass

            else:

                raise AssertionError("%s was accepted" % label)

            assert not (Path(tmp) / "faces" / "Alice").exists(), (
                "%s left a partial identity directory" % label
            )

            assert not (Path(tmp) / "cache.pkl").exists()


def test_bad_replacement_keeps_the_existing_identity():

    with tempfile.TemporaryDirectory() as tmp:

        enrollment = service(tmp)

        good = write_image(Path(tmp) / "good.png", 1)

        enrollment.enroll("Alice", [good])

        before = list_identity_images(Path(tmp) / "faces" / "Alice")

        assert len(before) == 1

        bad = write_image(Path(tmp) / "bad.png", 0)

        try:

            enrollment.enroll("Alice", [bad], replace=True)

        except EnrollmentError:

            pass

        else:

            raise AssertionError("a bad replacement was accepted")

        after = list_identity_images(Path(tmp) / "faces" / "Alice")

        assert [path.name for path in after] == [path.name for path in before]


def test_missing_and_undecodable_sources():

    with tempfile.TemporaryDirectory() as tmp:

        enrollment = service(tmp)

        try:

            enrollment.enroll("Alice", [Path(tmp) / "missing.png"])

        except EnrollmentError:

            pass

        else:

            raise AssertionError("a missing image was accepted")

        broken = Path(tmp) / "broken.png"

        broken.write_bytes(b"not an image")

        try:

            enrollment.enroll("Alice", [broken])

        except EnrollmentError:

            pass

        else:

            raise AssertionError("an undecodable image was accepted")

        try:

            collect_source_images([Path(tmp) / "empty"])

        except EnrollmentError:

            pass

        else:

            raise AssertionError("an empty source list was accepted")


# ----------------------------------------------------------------------
# D. Listing and removal
# ----------------------------------------------------------------------

def test_list_and_remove_identity():

    with tempfile.TemporaryDirectory() as tmp:

        faces = Path(tmp) / "faces"
        cache = FaceCache(str(Path(tmp) / "cache.pkl"))

        enrollment = service(tmp, cache)

        photo = write_image(Path(tmp) / "photo.png")

        enrollment.enroll("Alice", [photo])

        (faces / "EmptyFolder").mkdir()

        assert list_identities(faces) == ["Alice"]

        cached = cache.exists()

        assert cached

        removed = remove_identity("Alice", faces, cache)

        assert removed == faces / "Alice"
        assert not (faces / "Alice").exists()
        assert not cache.exists(), "removal left a stale embedding cache"

        try:

            remove_identity("Alice", faces, cache)

        except EnrollmentError:

            pass

        else:

            raise AssertionError("removing a missing identity was accepted")


# ----------------------------------------------------------------------
# E. Admin PIN
# ----------------------------------------------------------------------

def test_pin_hashing_and_verification():

    encoded = hash_pin("1234", iterations=100_000)

    assert encoded.startswith("pbkdf2_sha256$100000$")
    assert "1234" not in encoded
    assert verify_pin("1234", encoded) is True
    assert verify_pin("1235", encoded) is False
    assert verify_pin("1234", "") is False
    assert verify_pin("1234", "plain-text") is False

    # A different salt produces a different hash for the same PIN.
    second = hash_pin("1234", iterations=100_000)

    assert second != encoded
    assert verify_pin("1234", second) is True

    tampered = encoded[:-1] + ("A" if encoded[-1] != "A" else "B")

    assert verify_pin("1234", tampered) is False

    for bad in ("", "abc", "123", "1234567890123"):

        try:

            validate_pin(bad)

        except AdminAuthError:

            pass

        else:

            raise AssertionError("accepted invalid PIN %r" % (bad,))


def test_pin_config_round_trip():

    with tempfile.TemporaryDirectory() as tmp:

        path = Path(tmp) / "admin.yaml"

        assert load_pin_hash(path) == ""

        encoded = hash_pin("2468", iterations=100_000)

        save_pin_hash(path, encoded)

        text = path.read_text(encoding="utf-8")

        assert "2468" not in text
        assert load_pin_hash(path) == encoded
        assert verify_pin("2468", load_pin_hash(path)) is True

        try:

            save_pin_hash(path, "not-a-hash")

        except AdminAuthError:

            pass

        else:

            raise AssertionError("an invalid hash was written")


# ----------------------------------------------------------------------
# F. CLI
# ----------------------------------------------------------------------

def run_cli(args, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "enroll.py")] + args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_set_pin_and_gate():

    with tempfile.TemporaryDirectory() as tmp:

        admin = Path(tmp) / "admin.yaml"
        faces = Path(tmp) / "faces"

        environment = os.environ.copy()
        environment["GREETO_NEW_PIN"] = "2468"
        environment.pop("GREETO_ADMIN_PIN", None)

        result = run_cli(
            [
                "--admin-config",
                str(admin),
                "--faces-dir",
                str(faces),
                "set-pin",
            ],
            env=environment,
        )

        assert result.returncode == 0, result.stderr
        assert verify_pin("2468", load_pin_hash(admin))

        environment["GREETO_ADMIN_PIN"] = "0000"

        result = run_cli(
            [
                "--admin-config",
                str(admin),
                "--faces-dir",
                str(faces),
                "enroll",
                "--name",
                "Alice",
                "--images",
                str(Path(tmp) / "missing.png"),
            ],
            env=environment,
        )

        assert result.returncode == 1
        assert "incorrect admin pin" in result.stderr.lower(), result.stderr

        # The gate runs before the missing image is even inspected.
        assert "missing.png" not in result.stderr


def test_cli_list_and_visitors():

    with tempfile.TemporaryDirectory() as tmp:

        faces = Path(tmp) / "faces"
        faces.mkdir()

        write_image(faces / "Alice" / "1.png")
        (faces / "Empty").mkdir()

        result = run_cli(
            [
                "--faces-dir",
                str(faces),
                "list",
            ]
        )

        assert result.returncode == 0, result.stderr
        assert "Alice" in result.stdout
        assert "Empty" not in result.stdout

        database_path = Path(tmp) / "visitors.db"
        legacy = Path(tmp) / "visitors.json"
        legacy.write_text(
            '{"Alice": {"visits": 1162, "first_seen": "x", "last_seen": "y"}}',
            encoding="utf-8",
        )

        result = run_cli(
            [
                "visitors",
                "--database",
                str(database_path),
                "--legacy-json",
                str(legacy),
            ]
        )

        assert result.returncode == 0, result.stderr
        assert "Visitors: 1" in result.stdout, result.stdout
        assert "old visit counters" in result.stdout
        assert database_path.exists()
        assert not legacy.exists()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():

    print("Greeto enrolment and admin checks")

    check("A1. identity names are validated", test_identity_names)

    check("B1. valid enrolment writes images and cache",
          test_valid_enrolment_writes_images_and_cache)

    check("B2. duplicate source paths are collapsed",
          test_duplicate_source_paths_are_collapsed)

    check("B3. adding and replacing photos", test_add_and_replace)

    check("C1. bad batches write nothing", test_bad_batch_writes_nothing)

    check("C2. bad replacement keeps existing photos",
          test_bad_replacement_keeps_the_existing_identity)

    check("C3. missing and undecodable sources", test_missing_and_undecodable_sources)

    check("D1. listing and removal", test_list_and_remove_identity)

    check("E1. PIN hashing and verification", test_pin_hashing_and_verification)

    check("E2. PIN config round trip", test_pin_config_round_trip)

    check("F1. CLI set-pin and auth gate", test_cli_set_pin_and_gate)

    check("F2. CLI list and visitors", test_cli_list_and_visitors)

    print("-" * 74)

    if FAILURES:

        print("RESULT: %d FAILED -> %s" % (len(FAILURES), FAILURES))
        return 1

    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":

    sys.exit(main())
