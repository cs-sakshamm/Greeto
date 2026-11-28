"""Live webcam face enrolment and self-match verification.

This is the quick path for teaching the kiosk one person from the camera in
front of it, without first collecting a photo folder:

    python scripts/enroll_face.py --name "Saksham"
    python scripts/enroll_face.py --name "Saksham" --images photos/saksham
    python scripts/enroll_face.py --name "Saksham" --frames 5 --replace

Captured frames are validated with the exact same InsightFace preprocessing
and quality gates as :class:`VisionService`, then the pairwise cosine
similarity between the captured views is checked. A batch whose weakest
same-person pair is below 0.75 is rejected before anything is written:
those frames would sit too close to the kiosk's 0.45 recognition threshold
to be useful.

``scripts/enroll.py enroll`` remains the admin/PIN-protected path; this
helper is for local camera calibration and uses the same face store.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.exists():
    _VENV_PY = ROOT / ".venv" / "bin" / "python"

if _VENV_PY.exists():
    try:
        if Path(sys.executable).resolve() != _VENV_PY.resolve():
            try:
                import insightface  # noqa: F401
            except ImportError:
                import subprocess

                sys.exit(
                    subprocess.call(
                        [str(_VENV_PY), str(Path(__file__).resolve())]
                        + sys.argv[1:]
                    )
                )
    except Exception:
        pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from app.core.config_manager import ConfigManager  # noqa: E402
from app.vision.enrollment import (  # noqa: E402
    EnrollmentError,
    EnrollmentService,
    collect_source_images,
    validate_identity_name,
)
from app.vision.face_cache import FaceCache  # noqa: E402
from app.vision.recognition import RecognitionConfig  # noqa: E402
from app.vision.similarity import cosine_similarity  # noqa: E402


DEFAULT_FACES_DIRECTORY = ROOT / "models" / "faces"

DEFAULT_CACHE_FILE = ROOT / "models" / "cache" / "face_embeddings.pkl"

# A same-person pair below this is too variable for a live kiosk: a 0.45
# threshold with a 0.05 margin needs clearly separated reference views.
SELF_MATCH_FLOOR = 0.75


def _load_config() -> ConfigManager:

    manager = ConfigManager()

    manager.load()

    return manager


def _project_path(value: str | Path) -> Path:

    path = Path(value)

    if not path.is_absolute():

        path = ROOT / path

    return path.resolve()


def _camera_index(config: ConfigManager, override) -> int:

    if override is not None:

        return int(override)

    try:

        return int(
            config.get_value("camera", "camera", "index", default=0)
        )

    except (AttributeError, TypeError, ValueError):

        return 0


def _single_face_embedding(model, image, label: str):
    """Return the embedding of the only face in *image*, or raise."""

    faces = model.detect(image)

    if not faces:

        raise EnrollmentError("%s: no face detected" % label)

    if len(faces) > 1:

        raise EnrollmentError(
            "%s: %d faces detected; use a frame with one person"
            % (label, len(faces))
        )

    return faces[0].embedding


def _embedding_for_file(model, path: Path):

    image = cv2.imread(str(path))

    if image is None:

        raise EnrollmentError("could not decode image: %s" % path)

    return _single_face_embedding(model, image, path.name)


def capture_frames(
    model,
    config: RecognitionConfig,
    camera_index: int,
    count: int,
    warmup_frames: int = 10,
) -> list:
    """Capture *count* frames that contain exactly one usable face."""

    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():

        capture.release()

        raise EnrollmentError(
            "could not open camera %d; close other camera apps and retry"
            % camera_index
        )

    frames = []

    attempts = 0

    try:

        for _ in range(max(0, warmup_frames)):

            capture.read()

        while len(frames) < count and attempts < count * 20:

            attempts += 1

            ok, frame = capture.read()

            if not ok or frame is None:

                continue

            faces = model.detect(frame)

            if len(faces) != 1:

                continue

            face = faces[0]

            if float(face.confidence) < config.min_detection_score:

                continue

            if min(int(face.width), int(face.height)) < config.min_face_size:

                continue

            frames.append(frame)

            print(
                "  captured %d/%d (face %dx%d, detection %.2f)"
                % (len(frames), count, face.width, face.height, face.confidence)
            )

    finally:

        capture.release()

    if len(frames) < 2:

        raise EnrollmentError(
            "only %d usable frame(s); stand closer to the camera in even "
            "lighting and retry" % len(frames)
        )

    return frames


def _write_frames(frames, directory: Path) -> list[Path]:
    """Save captured frames as JPEGs so enrolment sees exactly these pixels."""

    directory.mkdir(parents=True, exist_ok=True)

    paths = []

    for index, frame in enumerate(frames, start=1):

        path = directory / ("%03d.jpg" % index)

        if not cv2.imwrite(str(path), frame):

            raise EnrollmentError("could not write %s" % path)

        paths.append(path)

    return paths


def self_match_scores(embeddings) -> tuple[float, float]:
    """Return ``(weakest, mean)`` pairwise cosine similarity."""

    scores = []

    for first in range(len(embeddings)):

        for second in range(first + 1, len(embeddings)):

            scores.append(
                cosine_similarity(embeddings[first], embeddings[second])
            )

    if not scores:

        raise EnrollmentError(
            "at least two images or captured frames are needed to verify "
            "a self-match"
        )

    return min(scores), sum(scores) / len(scores)


def verify_self_match(embeddings, floor: float = SELF_MATCH_FLOOR):
    """Reject a batch whose own views do not clearly look alike."""

    weakest, mean = self_match_scores(embeddings)

    print(
        "Self-match: weakest pair %.3f, mean %.3f (required > %.2f)"
        % (weakest, mean, floor)
    )

    if weakest <= floor:

        raise EnrollmentError(
            "self-match %.3f is at or below %.2f; recapture facing the "
            "camera in even lighting, or add clearer photos" % (weakest, floor)
        )

    return weakest, mean


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Enrol one person from the live webcam (or existing photos) and "
            "verify that the reference views match each other."
        )
    )

    parser.add_argument(
        "--name",
        required=True,
        help="identity name, also the folder name under models/faces",
    )

    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help=(
            "optional image files/directories; when omitted, frames are "
            "captured from the active webcam"
        ),
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="webcam frames to capture (default: 5, minimum 2)",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="camera index (default: config/camera.yaml camera.index)",
    )

    parser.add_argument(
        "--faces-dir",
        default=str(DEFAULT_FACES_DIRECTORY),
        help="face image root (default: models/faces)",
    )

    parser.add_argument(
        "--cache-file",
        default=str(DEFAULT_CACHE_FILE),
        help="embedding cache path",
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace all existing photos for this identity",
    )

    return parser


def command_enroll(args) -> int:

    name = validate_identity_name(args.name)

    if args.frames < 2:

        raise EnrollmentError("--frames must be at least 2 for verification")

    faces_directory = _project_path(args.faces_dir)
    cache = FaceCache(args.cache_file)

    manager = _load_config()

    config = RecognitionConfig.from_config(manager)

    for warning in config.warnings:

        print("Config warning: %s" % warning, file=sys.stderr)

    from app.vision.embedding_model import EmbeddingModel

    model = EmbeddingModel()

    staging = None

    try:

        if args.images:

            sources = collect_source_images(args.images)

            embeddings = [
                _embedding_for_file(model, path) for path in sources
            ]

            enroll_sources = sources

        else:

            print(
                "Capturing %d frames from camera %d..."
                % (args.frames, _camera_index(manager, args.camera))
            )

            frames = capture_frames(
                model,
                config,
                _camera_index(manager, args.camera),
                args.frames,
            )

            # The staging directory must live inside the face store:
            # enrolment moves the files with os.replace(), which cannot
            # cross drives on Windows.
            faces_directory.mkdir(parents=True, exist_ok=True)

            staging = Path(
                tempfile.mkdtemp(
                    prefix=".enroll-face-", dir=str(faces_directory)
                )
            )

            enroll_sources = _write_frames(frames, staging)

            embeddings = [
                _embedding_for_file(model, path) for path in enroll_sources
            ]

        verify_self_match(embeddings)

        service = EnrollmentService(
            model,
            faces_directory=faces_directory,
            cache=cache,
            config=config,
        )

        result = service.enroll(
            name,
            enroll_sources,
            replace=args.replace,
        )

    finally:

        if staging is not None:

            shutil.rmtree(staging, ignore_errors=True)

    print(
        "%s %s: added %d image(s), %d total."
        % (
            "Replaced" if result.replaced else "Enrolled",
            result.name,
            result.added,
            result.total,
        )
    )

    if result.cache_rebuilt:

        print("Embedding cache rebuilt.")

    else:

        print(
            "WARNING: %s"
            % (
                result.warning
                or "the embedding cache was not rebuilt; restarting will "
                "rebuild it from the images."
            ),
            file=sys.stderr,
        )

    try:

        confirmation_frames = int(
            manager.get_value(
                "vision", "recognition", "confirmation_frames", default=2
            )
        )

    except (AttributeError, TypeError, ValueError):

        confirmation_frames = 2

    print(
        "Recognition now uses threshold %.2f with a %.2f margin and %d "
        "confirmation frames."
        % (config.threshold, config.decision_margin, confirmation_frames)
    )
    print("Restart the kiosk so the running process sees the new identity.")
    print(
        "On the live camera, watch the 'Vision match:' log line to confirm "
        "the score for %s." % name
    )

    return 0


def main(argv=None) -> int:

    parser = build_parser()

    args = parser.parse_args(argv)

    try:

        return command_enroll(args)

    except EnrollmentError as error:

        print("Enrolment error: %s" % error, file=sys.stderr)
        return 1

    except KeyboardInterrupt:

        print("Cancelled.", file=sys.stderr)
        return 1

    except Exception as error:  # noqa: BLE001 - CLI must not traceback

        print("Error: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return 1


if __name__ == "__main__":

    sys.exit(main())
