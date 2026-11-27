"""
Admin enrolment CLI.

Examples:

    python scripts/enroll.py set-pin
    python scripts/enroll.py enroll --name "Asha Rao" --images photos/asha
    python scripts/enroll.py enroll --name "Asha Rao" --images a.jpg b.jpg
    python scripts/enroll.py list
    python scripts/enroll.py remove "Asha Rao"
    python scripts/enroll.py visitors

Mutating commands require the admin PIN. The PIN is read without echo
from the terminal, or from ``GREETO_ADMIN_PIN`` for non-interactive use.
New PINs for ``set-pin`` can be supplied through ``GREETO_NEW_PIN``.

The PIN itself is never written anywhere; only its salted PBKDF2 hash
in ``config/admin.yaml``.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
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
                sys.exit(subprocess.call([str(_VENV_PY), str(Path(__file__).resolve())] + sys.argv[1:]))
    except Exception:
        pass

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
from app.core.config_manager import ConfigManager  # noqa: E402
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
from app.vision.recognition import RecognitionConfig  # noqa: E402
from app.vision.visitor_database import VisitorDatabase  # noqa: E402


DEFAULT_ADMIN_CONFIG = ROOT / "config" / "admin.yaml"

DEFAULT_FACES_DIRECTORY = ROOT / "models" / "faces"

DEFAULT_CACHE_FILE = ROOT / "models" / "cache" / "face_embeddings.pkl"

MAX_PIN_ATTEMPTS = 3


def _read_secret(prompt: str) -> str:
    """Read a secret from the terminal, or fail clearly when headless."""

    if not sys.stdin.isatty():

        raise AdminAuthError(
            "no terminal available; set GREETO_ADMIN_PIN or run "
            "the command interactively"
        )

    try:

        return getpass.getpass(prompt)

    except (EOFError, KeyboardInterrupt) as error:

        raise AdminAuthError("PIN entry cancelled") from error


def authenticate(encoded: str, attempts: int = MAX_PIN_ATTEMPTS) -> None:
    """Require a valid PIN or raise AdminAuthError."""

    if not encoded:

        raise AdminAuthError(
            "no admin PIN configured; run: python scripts/enroll.py set-pin"
        )

    environment_pin = os.environ.get("GREETO_ADMIN_PIN")

    for attempt in range(attempts):

        if environment_pin is not None:

            pin = environment_pin

        else:

            pin = _read_secret("Admin PIN: ")

        if verify_pin(pin, encoded):

            return

        remaining = attempts - attempt - 1

        if environment_pin is not None or remaining <= 0:

            break

        print(
            "Incorrect PIN (%d attempt%s left)."
            % (remaining, "" if remaining == 1 else "s"),
            file=sys.stderr,
        )

    raise AdminAuthError("incorrect admin PIN")


def _read_new_pin() -> str:
    """Read and confirm a new numeric PIN."""

    environment_pin = os.environ.get("GREETO_NEW_PIN")

    if environment_pin is not None:

        return validate_pin(environment_pin)

    first = _read_secret("New admin PIN: ")

    second = _read_secret("Repeat new admin PIN: ")

    if first != second:

        raise AdminAuthError("the two PINs do not match")

    return validate_pin(first)


def _project_path(value: str | Path) -> Path:

    path = Path(value)

    if not path.is_absolute():

        path = ROOT / path

    return path.resolve()


def _admin_config_path(args) -> Path:

    return _project_path(args.admin_config)


def _faces_directory(args) -> Path:

    return _project_path(args.faces_dir)


def _cache(args) -> FaceCache:

    return FaceCache(args.cache_file)


def _load_config() -> ConfigManager:

    manager = ConfigManager()

    manager.load()

    return manager


def command_set_pin(args) -> int:
    """Create or rotate the admin PIN."""

    path = _admin_config_path(args)

    current = load_pin_hash(path)

    if current:

        authenticate(current)

    new_pin = _read_new_pin()

    save_pin_hash(path, hash_pin(new_pin))

    print("Admin PIN saved to %s" % path)

    print("The PIN itself was not stored.")

    return 0


def command_enroll(args) -> int:
    """Validate images, write the identity and rebuild the cache."""

    authenticate(load_pin_hash(_admin_config_path(args)))

    # Validate cheap inputs before loading the face model.
    validate_identity_name(args.name)

    sources = collect_source_images(args.images)

    # Imported here so list/remove/visitors do not load InsightFace.
    from app.vision.embedding_model import EmbeddingModel

    manager = _load_config()

    config = RecognitionConfig.from_config(manager)

    for warning in config.warnings:

        print("Config warning: %s" % warning, file=sys.stderr)

    model = EmbeddingModel()

    service = EnrollmentService(
        model,
        faces_directory=_faces_directory(args),
        cache=_cache(args),
        config=config,
    )

    result = service.enroll(
        args.name,
        sources,
        replace=args.replace,
    )

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
                or "the embedding cache was not rebuilt; restart will "
                "rebuild it from the images."
            ),
            file=sys.stderr,
        )

    print("Restart the kiosk so the running process sees the new identity.")

    return 0


def command_list(args) -> int:
    """List enrolled identity folders without loading the model."""

    names = list_identities(_faces_directory(args))

    if not names:

        print("No identities are enrolled.")
        print("Faces directory: %s" % _faces_directory(args))
        return 0

    print("Enrolled identities (%d):" % len(names))

    for name in names:

        image_count = len(
            list_identity_images(_faces_directory(args) / name)
        )

        print("  %-32s %d image(s)" % (name, image_count))

    return 0


def command_remove(args) -> int:
    """Remove an identity folder and invalidate the embedding cache."""

    authenticate(load_pin_hash(_admin_config_path(args)))

    removed = remove_identity(
        args.name,
        faces_directory=_faces_directory(args),
        cache=_cache(args),
    )

    print("Removed %s" % removed)
    print("The embedding cache was invalidated; it rebuilds on next start.")

    return 0


def command_visitors(args) -> int:
    """Show SQLite visitor statistics and run the legacy import once."""

    manager = _load_config()

    if args.database:

        database_path = args.database

        # An explicit temporary database must not read or archive the
        # real project data. Default to a sibling legacy file instead.
        legacy_path = args.legacy_json or str(
            Path(database_path).with_name("visitors.json")
        )

    else:

        database_path = manager.get_value(
            "vision",
            "visitors",
            "database_path",
            default="data/visitors.db",
        )

        legacy_path = args.legacy_json or manager.get_value(
            "vision",
            "visitors",
            "legacy_json",
            default="data/visitors.json",
        )

    database = VisitorDatabase(
        database_path=database_path,
        legacy_path=legacy_path,
    )

    database.initialize()

    stats = database.stats()

    print("Visitor database: %s" % stats["path"])
    print(
        "Visitors: %d    recorded visits: %d"
        % (stats["visitors"], stats["visits"])
    )

    migration = stats.get("migrated") or {}

    if migration.get("status") == "imported":

        print(
            "Legacy import: %s identities imported; old visit counters "
            "were discarded." % migration.get("imported", 0)
        )

        if migration.get("archived_to"):

            print("Legacy file archived to: %s" % migration["archived_to"])

    elif migration.get("status") == "invalid":

        print("Legacy file was invalid and was archived without import.")

    rows = database.list_visitors()

    if not rows:

        print("No visitor records yet.")
        return 0

    print("")
    print("%-28s %8s  %-19s  %-19s" % (
        "Name", "Visits", "First seen", "Last seen"
    ))

    for row in rows:

        print(
            "%-28s %8d  %-19s  %-19s"
            % (
                row["name"][:28],
                row["visits"],
                (row["first_seen"] or "-")[:19],
                (row["last_seen"] or "-")[:19],
            )
        )

    return 0


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Admin tools for Greeto face enrolment and visitor storage."
        )
    )

    parser.add_argument(
        "--admin-config",
        default=str(DEFAULT_ADMIN_CONFIG),
        help="admin PIN file (default: config/admin.yaml)",
    )

    parser.add_argument(
        "--faces-dir",
        default=str(DEFAULT_FACES_DIRECTORY),
        help="face image root (default: models/faces)",
    )

    parser.add_argument(
        "--cache-file",
        default=str(DEFAULT_CACHE_FILE),
        help="embedding cache path (default: models/cache/face_embeddings.pkl)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    set_pin = subparsers.add_parser(
        "set-pin",
        help="create or change the admin PIN",
    )

    set_pin.set_defaults(handler=command_set_pin)

    enroll = subparsers.add_parser(
        "enroll",
        help="enrol one identity from images",
    )

    enroll.add_argument(
        "--name",
        required=True,
        help="identity name, also the folder name under models/faces",
    )

    enroll.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="image files and/or directories containing jpg/jpeg/png",
    )

    enroll.add_argument(
        "--replace",
        action="store_true",
        help="replace all existing photos for this identity",
    )

    enroll.set_defaults(handler=command_enroll)

    list_parser = subparsers.add_parser(
        "list",
        help="list enrolled identities",
    )

    list_parser.set_defaults(handler=command_list)

    remove = subparsers.add_parser(
        "remove",
        help="remove an enrolled identity",
    )

    remove.add_argument("name", help="identity name to remove")

    remove.set_defaults(handler=command_remove)

    visitors = subparsers.add_parser(
        "visitors",
        help="show visitor statistics and run the one-time legacy import",
    )

    visitors.add_argument(
        "--database",
        default=None,
        help="override the visitor SQLite path",
    )

    visitors.add_argument(
        "--legacy-json",
        default=None,
        help="override the legacy JSON path",
    )

    visitors.set_defaults(handler=command_visitors)

    return parser


def main(argv=None) -> int:

    parser = build_parser()

    args = parser.parse_args(argv)

    try:

        return args.handler(args)

    except AdminAuthError as error:

        print("Admin error: %s" % error, file=sys.stderr)
        return 1

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
