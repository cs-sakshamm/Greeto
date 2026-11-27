"""
Phase 3 Milestone 3 visitor storage checks.

These use temporary directories only: no project data, camera, model or
GUI is touched.

    python tests/test_storage.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

from app.vision.visitor_database import (  # noqa: E402
    SCHEMA_VERSION,
    VisitorDatabase,
    normalize_visitor_name,
)


FAILURES = []


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


def make_database(tmp, legacy=None):
    """Create a database rooted entirely inside the temporary directory."""

    return VisitorDatabase(
        database_path=Path(tmp) / "visitors.db",
        legacy_path=(
            Path(legacy)
            if legacy is not None
            else Path(tmp) / "visitors.json"
        ),
    )


# ----------------------------------------------------------------------
# A. Basic persistence
# ----------------------------------------------------------------------

def test_new_database_starts_empty():

    with tempfile.TemporaryDirectory() as tmp:

        database = make_database(tmp)

        stats = database.stats()

        assert stats["exists"] is False
        assert stats["visitors"] == 0
        assert stats["visits"] == 0

        # Constructing does not create anything until first use.
        assert not database.database_path.exists()

        assert database.get("Alice") is None

        assert database.database_path.exists()

        record = database.register("Alice")

        assert record["name"] == "Alice"
        assert record["visits"] == 1
        assert record["first_seen"]
        assert record["last_seen"]


def test_repeated_visits_are_counted():

    with tempfile.TemporaryDirectory() as tmp:

        database = make_database(tmp)

        database.register("Alice", seen_at="2026-01-01T10:00:00")
        database.register("Bob", seen_at="2026-01-02T10:00:00")
        final = database.register(
            "Alice", seen_at="2026-01-03T10:00:00"
        )

        assert final["visits"] == 2
        assert final["first_seen"] == "2026-01-01T10:00:00"
        assert final["last_seen"] == "2026-01-03T10:00:00"

        assert database.get("Alice")["visits"] == 2
        assert database.get("Bob")["visits"] == 1
        assert database.get("Nobody") is None

        stats = database.stats()

        assert stats["visitors"] == 2
        assert stats["visits"] == 3


def test_persistence_across_instances():

    with tempfile.TemporaryDirectory() as tmp:

        first = make_database(tmp)

        first.register("Alice")

        second = make_database(tmp)

        assert second.get("alice")["visits"] == 1

        second.register("ALICE")

        assert second.get("Alice")["visits"] == 2

        rows = second.list_visitors()

        assert len(rows) == 1, rows
        assert rows[0]["name"] == "Alice", (
            "the original display spelling was replaced"
        )


def test_names_are_validated_and_tidied():

    assert normalize_visitor_name("  Asha   Rao  ") == "Asha Rao"

    with tempfile.TemporaryDirectory() as tmp:

        database = make_database(tmp)

        for bad in ("", "   ", None, 5, "line\nbreak", "x" * 129):

            try:

                database.register(bad)

            except ValueError:

                pass

            else:

                raise AssertionError("accepted invalid name %r" % (bad,))


# ----------------------------------------------------------------------
# B. Legacy import
# ----------------------------------------------------------------------

def test_legacy_import_drops_old_statistics():

    with tempfile.TemporaryDirectory() as tmp:

        legacy = Path(tmp) / "visitors.json"

        legacy.write_text(
            json.dumps(
                {
                    "Alice": {
                        "first_seen": "2026-07-01T10:00:00",
                        "visits": 1162,
                        "last_seen": "2026-11-01T10:00:00",
                    },
                    "  Bob  ": {
                        "first_seen": "2026-07-02T10:00:00",
                        "visits": 5,
                        "last_seen": "2026-11-02T10:00:00",
                    },
                }
            ),
            encoding="utf-8",
        )

        database = make_database(tmp, legacy)

        database.initialize()

        stats = database.stats()

        assert stats["visitors"] == 2
        assert stats["visits"] == 0, "old visit counters were imported"

        for name in ("Alice", "Bob"):

            record = database.get(name)

            assert record["visits"] == 0
            assert record["first_seen"] is None
            assert record["last_seen"] is None
            assert record["legacy_import"] is True

        archives = list(Path(tmp).glob("visitors.json.migrated*"))

        assert len(archives) == 1, archives

        assert not legacy.exists()

        # A second instance must not import the archive again.
        second = make_database(tmp, legacy)

        second.initialize()

        assert second.stats()["visitors"] == 2

        second.register("Alice")

        assert second.get("Alice")["visits"] == 1


def test_invalid_legacy_file_is_archived_safely():

    with tempfile.TemporaryDirectory() as tmp:

        legacy = Path(tmp) / "visitors.json"

        legacy.write_text("{ this is not json", encoding="utf-8")

        database = make_database(tmp, legacy)

        database.initialize()

        assert database.stats()["visitors"] == 0

        archives = list(Path(tmp).glob("visitors.json.invalid*"))

        assert len(archives) == 1, archives

        assert not legacy.exists()

        # The database still works.
        database.register("Alice")

        assert database.get("Alice")["visits"] == 1


# ----------------------------------------------------------------------
# C. Recovery and schema safety
# ----------------------------------------------------------------------

def test_corrupt_database_is_archived():

    with tempfile.TemporaryDirectory() as tmp:

        path = Path(tmp) / "visitors.db"

        path.write_bytes(b"this is definitely not a sqlite database")

        database = make_database(tmp)

        database.initialize()

        archives = list(Path(tmp).glob("visitors.db.corrupt-*"))

        assert len(archives) == 1, archives

        assert archives[0].read_bytes().startswith(b"this is definitely")

        database.register("Alice")

        assert database.get("Alice")["visits"] == 1


def test_newer_schema_is_refused():

    with tempfile.TemporaryDirectory() as tmp:

        path = Path(tmp) / "visitors.db"

        connection = sqlite3.connect(str(path))

        try:

            connection.execute(
                "PRAGMA user_version = %d" % (SCHEMA_VERSION + 1)
            )

            connection.commit()

        finally:

            connection.close()

        database = make_database(tmp)

        try:

            database.initialize()

        except RuntimeError as error:

            assert "newer" in str(error)

        else:

            raise AssertionError("a newer database schema was accepted")


def test_temporary_database_does_not_touch_project_data():

    # This is the contract the CLI relies on when --database is used.
    with tempfile.TemporaryDirectory() as tmp:

        database = make_database(tmp)

        database.initialize()

        assert database.legacy_path == Path(tmp) / "visitors.json"

        assert database.database_path.parent == Path(tmp)


# ----------------------------------------------------------------------
# D. Concurrency
# ----------------------------------------------------------------------

def test_concurrent_registration():

    with tempfile.TemporaryDirectory() as tmp:

        database = make_database(tmp)

        names = ["Person %d" % index for index in range(10)]

        errors = []

        def worker(name, index):

            try:

                database.register(name, seen_at="2026-01-01T00:00:%02d" % index)

            except Exception as error:  # noqa: BLE001

                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(name, index))
            for index, name in enumerate(names * 4)
        ]

        for thread in threads:

            thread.start()

        for thread in threads:

            thread.join()

        assert not errors, errors

        assert database.stats()["visits"] == 40

        for name in names:

            assert database.get(name)["visits"] == 4

        print("       40 writes from 40 threads, no lost visits")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():

    print("Greeto visitor storage checks")

    check("A1. empty database is created lazily", test_new_database_starts_empty)

    check("A2. repeated visits are counted", test_repeated_visits_are_counted)

    check("A3. records persist across instances", test_persistence_across_instances)

    check("A4. names are validated and tidied", test_names_are_validated_and_tidied)

    check("B1. legacy identities import, old stats discarded",
          test_legacy_import_drops_old_statistics)

    check("B2. invalid legacy file is archived", test_invalid_legacy_file_is_archived_safely)

    check("C1. corrupt database is archived", test_corrupt_database_is_archived)

    check("C2. newer schema is refused", test_newer_schema_is_refused)

    check("C3. temporary database is isolated", test_temporary_database_does_not_touch_project_data)

    check("D1. concurrent registration loses no visits", test_concurrent_registration)

    print("-" * 74)

    if FAILURES:

        print("RESULT: %d FAILED -> %s" % (len(FAILURES), FAILURES))
        return 1

    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":

    sys.exit(main())
