"""
Visitor Database.

Stores visitor statistics in SQLite instead of rewriting a JSON file on
every arrival. Two tables are used:

``visitors``  one row per person, with first/last seen timestamps
``visits``    one row per announced visit

The legacy ``data/visitors.json`` file is read once. Only the identities
are imported; its visit counters and timestamps are discarded because
the old code incremented the counter on every detection cycle, so those
numbers are an artefact rather than real traffic. The JSON file is then
archived next to itself so it is never imported twice.

A corrupt database is archived and replaced with a fresh one instead of
preventing the kiosk from starting. Initialisation is lazy: constructing
this object has no filesystem side effects until a visit is recorded or
a read is requested.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


# Repository root, independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_PATH = "data/visitors.db"

DEFAULT_LEGACY_PATH = "data/visitors.json"

SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS visitors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
    first_seen TEXT,
    last_seen TEXT,
    legacy_import INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id INTEGER NOT NULL REFERENCES visitors(id) ON DELETE CASCADE,
    seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_visits_visitor
    ON visits(visitor_id, seen_at);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def normalize_visitor_name(name: str) -> str:
    """Validate and tidy a visitor name.

    Names are display strings, not paths, so the rules are deliberately
    permissive: collapse whitespace, reject empty values, control
    characters and unreasonable lengths.
    """

    if not isinstance(name, str):

        raise ValueError("visitor name must be a string")

    if any(
        ord(character) < 32 or ord(character) == 127
        for character in name
    ):

        raise ValueError("visitor name contains control characters")

    cleaned = " ".join(name.split())

    if not cleaned:

        raise ValueError("visitor name is empty")

    if len(cleaned) > 128:

        raise ValueError("visitor name is longer than 128 characters")

    return cleaned


def _resolve(path: str | Path) -> Path:

    path = Path(path)

    if not path.is_absolute():

        path = PROJECT_ROOT / path

    return path


def _now() -> str:

    return datetime.now().isoformat(timespec="seconds")


def _unique_path(path: Path) -> Path:

    if not path.exists():

        return path

    stem = path.name

    for index in range(1, 1000):

        candidate = path.with_name("%s.%d" % (stem, index))

        if not candidate.exists():

            return candidate

    raise OSError("could not find a free archive name for %s" % path)


class VisitorDatabase:
    """Persistent, thread-safe visitor storage backed by SQLite."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        legacy_path: str | Path | None = None,
    ) -> None:

        self.database_path = _resolve(
            database_path or DEFAULT_DATABASE_PATH
        )

        if legacy_path is None:

            # Keep the legacy file next to the database by default, so a
            # temporary database never reads the real project data.
            legacy_path = self.database_path.parent / "visitors.json"

        self.legacy_path = _resolve(legacy_path)

        self._lock = threading.RLock()

        self._initialized = False

        self._last_migration: dict[str, object] = {}

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    def initialize(self) -> "VisitorDatabase":
        """Create the schema and run the one-time legacy import."""

        with self._lock:

            if self._initialized:

                return self

            self.database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._create_schema_with_recovery()

            self._last_migration = self._migrate_legacy()

            self._initialized = True

        return self

    def close(self) -> None:
        """Compatibility no-op; connections are short-lived."""

    # -----------------------------------------------------
    # Writes
    # -----------------------------------------------------

    def register(
        self,
        name: str,
        seen_at: str | None = None,
    ) -> dict:
        """Record one visit and return the updated visitor record."""

        cleaned = normalize_visitor_name(name)

        timestamp = seen_at or _now()

        with self._lock:

            self.initialize()

            connection = self._connect()

            try:

                connection.execute("BEGIN IMMEDIATE")

                row = connection.execute(
                    "SELECT id FROM visitors WHERE name = ? COLLATE NOCASE",
                    (cleaned,),
                ).fetchone()

                if row is None:

                    cursor = connection.execute(
                        "INSERT INTO visitors(name, first_seen, last_seen) "
                        "VALUES (?, ?, ?)",
                        (cleaned, timestamp, timestamp),
                    )

                    visitor_id = int(cursor.lastrowid)

                else:

                    visitor_id = int(row["id"])

                    connection.execute(
                        "UPDATE visitors SET last_seen = ? WHERE id = ?",
                        (timestamp, visitor_id),
                    )

                connection.execute(
                    "INSERT INTO visits(visitor_id, seen_at) VALUES (?, ?)",
                    (visitor_id, timestamp),
                )

                connection.commit()

            except Exception:

                connection.rollback()

                raise

            finally:

                connection.close()

            record = self.get(cleaned)

            if record is None:

                raise RuntimeError("visitor record disappeared after write")

            return record

    # -----------------------------------------------------
    # Reads
    # -----------------------------------------------------

    def get(self, name: str) -> dict | None:
        """Return one visitor record, or None when the person is unknown."""

        cleaned = normalize_visitor_name(name)

        with self._lock:

            self.initialize()

            connection = self._connect()

            try:

                row = connection.execute(
                    "SELECT v.id, v.name, v.first_seen, v.last_seen, "
                    "v.legacy_import, "
                    "(SELECT COUNT(*) FROM visits WHERE visitor_id = v.id) "
                    "AS visits "
                    "FROM visitors AS v "
                    "WHERE v.name = ? COLLATE NOCASE",
                    (cleaned,),
                ).fetchone()

            finally:

                connection.close()

            return self._to_record(row) if row is not None else None

    def list_visitors(self, limit: int | None = None) -> list[dict]:
        """Return visitors, most recently seen first."""

        with self._lock:

            self.initialize()

            query = (
                "SELECT v.id, v.name, v.first_seen, v.last_seen, "
                "v.legacy_import, "
                "(SELECT COUNT(*) FROM visits WHERE visitor_id = v.id) "
                "AS visits "
                "FROM visitors AS v "
                "ORDER BY COALESCE(v.last_seen, '') DESC, v.name "
            )

            parameters: tuple = ()

            if limit is not None:

                query += "LIMIT ?"

                parameters = (max(0, int(limit)),)

            connection = self._connect()

            try:

                rows = connection.execute(query, parameters).fetchall()

            finally:

                connection.close()

            return [self._to_record(row) for row in rows]

    def stats(self) -> dict:
        """Small summary for diagnostics and the admin CLI."""

        if not self._initialized and not self.database_path.exists():

            return {
                "path": str(self.database_path),
                "exists": False,
                "visitors": 0,
                "visits": 0,
                "migrated": self._last_migration,
            }

        with self._lock:

            self.initialize()

            connection = self._connect()

            try:

                visitors = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM visitors"
                    ).fetchone()[0]
                )

                visits = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM visits"
                    ).fetchone()[0]
                )

            finally:

                connection.close()

        return {
            "path": str(self.database_path),
            "exists": True,
            "visitors": visitors,
            "visits": visits,
            "migrated": dict(self._last_migration),
        }

    # -----------------------------------------------------
    # SQLite helpers
    # -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:

        connection = sqlite3.connect(
            str(self.database_path),
            timeout=5.0,
        )

        connection.row_factory = sqlite3.Row

        try:

            connection.execute("PRAGMA foreign_keys = ON")

            connection.execute("PRAGMA busy_timeout = 5000")

            connection.execute("PRAGMA journal_mode = WAL")

        except Exception:

            connection.close()

            raise

        return connection

    def _create_schema_with_recovery(self) -> None:

        try:

            self._create_schema()

            return

        except sqlite3.DatabaseError as error:

            if not self._is_corrupt_database_file():

                raise

            archived = self._archive_database()

            logger.error(
                "Visitor database was unreadable (%s); archived to %s "
                "and created a fresh one.",
                error,
                archived,
            )

        self._create_schema()

    def _create_schema(self) -> None:

        connection = self._connect()

        try:

            version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )

            if version > SCHEMA_VERSION:

                raise RuntimeError(
                    "visitor database schema version %d is newer than "
                    "this application supports (%d)"
                    % (version, SCHEMA_VERSION)
                )

            connection.executescript(SCHEMA)

            if version < SCHEMA_VERSION:

                connection.execute(
                    "PRAGMA user_version = %d" % SCHEMA_VERSION
                )

            connection.commit()

        finally:

            connection.close()

    def _is_corrupt_database_file(self) -> bool:

        if not self.database_path.exists():

            return False

        try:

            with self.database_path.open("rb") as file:

                header = file.read(16)

        except OSError:

            return False

        # An empty file is a valid new database; a non-empty file that
        # does not start with the SQLite signature cannot be recovered.
        return bool(header) and header != b"SQLite format 3\x00"

    def _archive_database(self) -> Path:

        archive = _unique_path(
            self.database_path.with_name(
                "%s.corrupt-%s"
                % (
                    self.database_path.name,
                    datetime.now().strftime("%Y%m%dT%H%M%S"),
                )
            )
        )

        os.replace(self.database_path, archive)

        for suffix in ("-wal", "-shm"):

            sidecar = Path(str(self.database_path) + suffix)

            if sidecar.exists():

                os.replace(sidecar, Path(str(archive) + suffix))

        return archive

    # -----------------------------------------------------
    # Legacy JSON import
    # -----------------------------------------------------

    def _migrate_legacy(self) -> dict[str, object]:

        if not self.legacy_path.exists():

            return {
                "status": "absent",
                "imported": 0,
                "archived_to": None,
            }

        try:

            data = json.loads(
                self.legacy_path.read_text(encoding="utf-8")
            )

        except (OSError, ValueError) as error:

            logger.warning(
                "Legacy visitor file could not be read (%s); archiving it "
                "without importing anything.",
                error,
            )

            data = {}

            invalid = True

        else:

            invalid = not isinstance(data, dict)

            if invalid:

                logger.warning(
                    "Legacy visitor file does not contain a mapping; "
                    "archiving it without importing anything."
                )

                data = {}

        imported = 0

        connection = self._connect()

        try:

            if data:

                connection.execute("BEGIN IMMEDIATE")

                for raw_name in data:

                    try:

                        name = normalize_visitor_name(raw_name)

                    except ValueError:

                        continue

                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO visitors"
                        "(name, first_seen, last_seen, legacy_import) "
                        "VALUES (?, NULL, NULL, 1)",
                        (name,),
                    )

                    imported += max(0, cursor.rowcount)

                connection.commit()

        except Exception:

            connection.rollback()

            raise

        finally:

            connection.close()

        archive = _unique_path(
            self.legacy_path.with_name(
                self.legacy_path.name
                + (".invalid" if invalid else ".migrated")
            )
        )

        archived_to: str | None = None

        try:

            os.replace(self.legacy_path, archive)

            archived_to = str(archive)

        except OSError as error:

            logger.warning(
                "Legacy visitor file could not be archived (%s); it will "
                "not be imported again.",
                error,
            )

        self._set_metadata(
            "legacy_import",
            {
                "status": "invalid" if invalid else "imported",
                "imported": imported,
                "source": str(self.legacy_path),
                "archived_to": archived_to,
                "imported_at": _now(),
            },
        )

        return {
            "status": "invalid" if invalid else "imported",
            "imported": imported,
            "archived_to": archived_to,
        }

    def _set_metadata(self, key: str, payload: dict) -> None:

        connection = self._connect()

        try:

            connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(payload, sort_keys=True)),
            )

            connection.commit()

        finally:

            connection.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> dict:

        return {
            "name": row["name"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "visits": int(row["visits"]),
            "legacy_import": bool(row["legacy_import"]),
        }
