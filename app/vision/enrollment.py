"""
Face enrolment.

The kiosk learns people from image folders under ``models/faces``. This
module performs the offline, admin-only part of that workflow:

* every supplied image is decoded and must contain exactly one usable
  face (quality gates are the same as recognition);
* all images are validated before anything is written, so a bad batch
  cannot leave a half-enrolled identity;
* images are copied as decoded pixels, not as arbitrary file bytes;
* the embedding cache is deleted and rebuilt so the next kiosk start
  sees the new identity.

This module does not handle PINs or command-line arguments. Those live in
``scripts/enroll.py``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.vision.face_cache import FaceCache
from app.vision.face_database import FaceDatabase
from app.vision.models import Face
from app.vision.recognition import RecognitionConfig
from app.vision.similarity import normalize_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FACES_DIRECTORY = "models/faces"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(("COM%d" % index) for index in range(1, 10)),
    *(("LPT%d" % index) for index in range(1, 10)),
}

INVALID_NAME_CHARACTERS = set('<>:"/\\|?*')


class EnrollmentError(ValueError):
    """Raised when an identity or image batch cannot be enrolled."""


@dataclass(frozen=True)
class EnrollmentResult:
    """Outcome of one successful enrolment."""

    name: str

    added: int

    total: int

    replaced: bool

    cache_rebuilt: bool

    warning: str = ""


def validate_identity_name(name: str) -> str:
    """Return a safe folder/display name, or raise EnrollmentError."""

    if not isinstance(name, str):

        raise EnrollmentError("identity name must be text")

    if any(
        ord(character) < 32 or ord(character) == 127
        for character in name
    ):

        raise EnrollmentError(
            "identity name contains control characters"
        )

    cleaned = " ".join(name.split())

    if not cleaned:

        raise EnrollmentError("identity name is empty")

    if len(cleaned) > 64:

        raise EnrollmentError("identity name is longer than 64 characters")

    if cleaned in (".", "..") or cleaned.startswith("."):

        raise EnrollmentError(
            "identity name cannot start with a dot"
        )

    if any(character in INVALID_NAME_CHARACTERS for character in cleaned):

        raise EnrollmentError(
            "identity name contains a path or reserved character"
        )

    if cleaned.split(".")[0].upper() in WINDOWS_RESERVED_NAMES:

        raise EnrollmentError(
            "identity name is reserved by the operating system"
        )

    if Path(cleaned).name != cleaned:

        raise EnrollmentError("identity name is not a plain folder name")

    return cleaned


def _resolve_directory(path: str | Path) -> Path:

    path = Path(path)

    if not path.is_absolute():

        path = PROJECT_ROOT / path

    return path


def list_identity_images(directory: str | Path) -> list[Path]:
    """Return supported image files in ``directory``, sorted."""

    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():

        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def list_identities(faces_directory: str | Path) -> list[str]:
    """List identity folders that contain at least one image."""

    faces_directory = _resolve_directory(faces_directory)

    if not faces_directory.exists():

        return []

    return sorted(
        path.name
        for path in faces_directory.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and list_identity_images(path)
    )


def collect_source_images(paths) -> list[Path]:
    """Expand files and directories into a de-duplicated image list."""

    if isinstance(paths, (str, Path)):

        paths = [paths]

    collected: list[Path] = []

    seen: set[Path] = set()

    for raw_path in paths:

        path = Path(raw_path)

        if not path.exists():

            raise EnrollmentError("image path does not exist: %s" % path)

        if path.is_dir():

            candidates = list_identity_images(path)

        elif path.is_file():

            candidates = [path]

        else:

            raise EnrollmentError("unsupported image path: %s" % path)

        for candidate in candidates:

            if candidate.suffix.lower() not in IMAGE_SUFFIXES:

                raise EnrollmentError(
                    "unsupported image type: %s" % candidate
                )

            resolved = candidate.resolve()

            if resolved in seen:

                continue

            seen.add(resolved)

            collected.append(candidate)

    if not collected:

        raise EnrollmentError("no .jpg, .jpeg or .png images found")

    return collected


def _next_image_index(identity_directory: Path) -> int:

    highest = 0

    for image in list_identity_images(identity_directory):

        stem = image.stem

        if stem.isdigit():

            highest = max(highest, int(stem))

    return highest + 1


def _check_face(
    face: Face,
    config: RecognitionConfig,
) -> None:
    """Apply the recognition quality gates to an enrolment face."""

    width = int(face.width)

    height = int(face.height)

    confidence = float(face.confidence)

    if width <= 0 or height <= 0:

        raise EnrollmentError(
            "detected face has an invalid bounding box"
        )

    if confidence < config.min_detection_score:

        raise EnrollmentError(
            "detection score %.3f is below %.3f"
            % (confidence, config.min_detection_score)
        )

    if min(width, height) < config.min_face_size:

        raise EnrollmentError(
            "face is %dx%d; minimum side is %d"
            % (width, height, config.min_face_size)
        )

    try:

        normalize_embedding(
            face.embedding,
            expected_size=config.expected_embedding_size,
            min_norm=config.min_embedding_norm,
            normalize=config.normalize_embeddings,
        )

    except ValueError as error:

        raise EnrollmentError(
            "face embedding is invalid: %s" % error
        ) from error


class EnrollmentService:
    """Validate images and add them to the face database."""

    def __init__(
        self,
        model,
        faces_directory: str | Path = DEFAULT_FACES_DIRECTORY,
        cache: FaceCache | None = None,
        config: RecognitionConfig | None = None,
    ) -> None:

        self.model = model

        self.faces_directory = _resolve_directory(faces_directory)

        self.cache = cache if cache is not None else FaceCache()

        self.config = config if config is not None else RecognitionConfig()

        if not isinstance(self.config, RecognitionConfig):

            raise TypeError("config must be a RecognitionConfig")

    # -----------------------------------------------------

    def enroll(
        self,
        name: str,
        image_paths,
        *,
        replace: bool = False,
    ) -> EnrollmentResult:
        """Enrol one identity from images, or raise without writing."""

        name = validate_identity_name(name)

        sources = collect_source_images(image_paths)

        prepared: list[tuple[Path, object]] = []

        for source in sources:

            image = cv2.imread(str(source))

            if image is None:

                raise EnrollmentError(
                    "could not decode image: %s" % source
                )

            faces = self.model.detect(image)

            if not faces:

                raise EnrollmentError(
                    "no face found in %s" % source.name
                )

            if len(faces) > 1:

                raise EnrollmentError(
                    "%s contains %d faces; use an image with one face"
                    % (source.name, len(faces))
                )

            try:

                _check_face(faces[0], self.config)

            except EnrollmentError as error:

                raise EnrollmentError(
                    "%s: %s" % (source.name, error)
                ) from error

            prepared.append((source, image))

        self.faces_directory.mkdir(parents=True, exist_ok=True)

        identity_directory = self.faces_directory / name

        if identity_directory.exists() and not identity_directory.is_dir():

            raise EnrollmentError(
                "identity path exists but is not a directory: %s"
                % identity_directory
            )

        if identity_directory.is_symlink():

            raise EnrollmentError(
                "identity directory must not be a symbolic link"
            )

        if (
            self.faces_directory.resolve()
            not in identity_directory.resolve().parents
        ):

            raise EnrollmentError(
                "identity directory escapes the face database"
            )

        staging = Path(
            tempfile.mkdtemp(
                prefix=".enroll-",
                dir=str(self.faces_directory),
            )
        )

        try:

            for index, (source, image) in enumerate(prepared, start=1):

                output = staging / ("%03d.jpg" % index)

                if not cv2.imwrite(str(output), image):

                    raise EnrollmentError(
                        "could not write image for %s" % source.name
                    )

            if replace:

                self._replace_identity(identity_directory, staging)

            else:

                self._merge_identity(identity_directory, staging)

        except Exception:

            shutil.rmtree(staging, ignore_errors=True)

            raise

        total = len(list_identity_images(identity_directory))

        cache_rebuilt, warning = self._rebuild_cache()

        return EnrollmentResult(
            name=name,
            added=len(prepared),
            total=total,
            replaced=bool(replace),
            cache_rebuilt=cache_rebuilt,
            warning=warning,
        )

    # -----------------------------------------------------

    def _replace_identity(
        self,
        identity_directory: Path,
        staging: Path,
    ) -> None:

        if not identity_directory.exists():

            os.replace(staging, identity_directory)

            return

        backup = Path(
            tempfile.mkdtemp(
                prefix=(".%s-old-" % identity_directory.name),
                dir=str(self.faces_directory),
            )
        )

        # mkdtemp created an empty directory; replace() needs it gone.
        backup.rmdir()

        os.replace(identity_directory, backup)

        try:

            os.replace(staging, identity_directory)

        except Exception:

            if not identity_directory.exists():

                os.replace(backup, identity_directory)

            raise

        else:

            shutil.rmtree(backup, ignore_errors=True)

    def _merge_identity(
        self,
        identity_directory: Path,
        staging: Path,
    ) -> None:

        if not identity_directory.exists():

            os.replace(staging, identity_directory)

            return

        next_index = _next_image_index(identity_directory)

        for staged in sorted(staging.iterdir()):

            if not staged.is_file():

                continue

            destination = identity_directory / (
                "%03d.jpg" % next_index
            )

            while destination.exists():

                next_index += 1

                destination = identity_directory / (
                    "%03d.jpg" % next_index
                )

            os.replace(staged, destination)

            next_index += 1

        shutil.rmtree(staging, ignore_errors=True)

    # -----------------------------------------------------

    def _rebuild_cache(self) -> tuple[bool, str]:

        try:

            self.cache.delete()

        except OSError as error:

            return False, "could not delete the old cache: %s" % error

        try:

            FaceDatabase(
                self.model,
                database_path=self.faces_directory,
                cache=self.cache,
            ).load()

        except Exception as error:  # noqa: BLE001 - enrolment already succeeded

            try:

                self.cache.delete()

            except OSError:

                pass

            return (
                False,
                "images were saved but the cache could not be "
                "rebuilt: %s" % error,
            )

        if not self.cache.exists():

            return False, "images were saved but the cache was not written"

        return True, ""


def remove_identity(
    name: str,
    faces_directory: str | Path = DEFAULT_FACES_DIRECTORY,
    cache: FaceCache | None = None,
) -> Path:
    """Delete one identity folder and invalidate the embedding cache."""

    name = validate_identity_name(name)

    faces_directory = _resolve_directory(faces_directory)

    if cache is None:

        cache = FaceCache()

    identity_directory = faces_directory / name

    if not identity_directory.exists():

        raise EnrollmentError("identity is not enrolled: %s" % name)

    if identity_directory.is_symlink():

        raise EnrollmentError("refusing to remove a symbolic link")

    if identity_directory.resolve().parent != faces_directory.resolve():

        raise EnrollmentError("identity directory escapes the face database")

    shutil.rmtree(identity_directory)

    if cache is not None:

        try:

            cache.delete()

        except OSError as error:

            raise EnrollmentError(
                "identity removed, but the cache could not be "
                "invalidated: %s" % error
            ) from error

    return identity_directory
