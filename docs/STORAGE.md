# Visitor Storage and Enrolment

Phase 3, Milestone 3.

This document covers the SQLite visitor database and the offline admin
tools that enrol people into `models/faces`.

## 1. Visitor database

`app/vision/visitor_database.py` replaces the old JSON file with a small
SQLite database at:

```
data/visitors.db
```

The path and the legacy import path are configurable in
`config/vision.yaml`:

```yaml
visitors:
  database_path: data/visitors.db
  legacy_json: data/visitors.json
```

Relative paths resolve against the project root, not the current
working directory.

### Schema

```sql
visitors (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE COLLATE NOCASE,
    first_seen TEXT,
    last_seen TEXT,
    legacy_import INTEGER
)

visits (
    id INTEGER PRIMARY KEY,
    visitor_id INTEGER REFERENCES visitors(id) ON DELETE CASCADE,
    seen_at TEXT
)
```

A visitor's `visits` count is calculated from the `visits` table, so it
cannot drift from the visit rows. The kiosk records one visit when a
tracked visitor is announced, not on every processed frame. Unknown
faces are not stored.

SQLite runs in WAL mode with a five-second busy timeout. Each operation
uses a short-lived connection and a process-local lock; a visit insert
updates the visitor row and inserts the visit row in one transaction.

### Legacy import

On first use, an existing `data/visitors.json` is read once:

- every valid identity name is imported as a visitor with no visits;
- `visits`, `first_seen` and `last_seen` from the old file are
  **discarded**. The old code incremented `visits` on every detection
  cycle, so values such as `1162` were an artefact, not traffic;
- the JSON file is moved to a sibling `.migrated` archive, so it cannot
  be imported twice;
- a malformed file is archived as `.invalid` and no names are imported.

Imported people appear in the admin list with zero visits and no
timestamps until their next real visit. A backup of the old file remains
next to the database.

### Recovery

- A corrupt/non-SQLite database file is renamed to
  `visitors.db.corrupt-<timestamp>` and a fresh database is created.
- A database with a newer schema version is refused rather than
  modified.
- Constructing `VisitorDatabase` does not create files. Initialisation
  happens on the first read or write, so tests and imports are free of
  side effects.
- A storage failure while recording a visit is logged and never stops
  the greeting pipeline.

## 2. Admin PIN

Mutating CLI commands require a numeric admin PIN of 4-12 digits. The
PIN is never stored or written to logs. `config/admin.yaml` contains
only a salted PBKDF2-HMAC-SHA256 hash:

```yaml
admin:
  pin_hash: pbkdf2_sha256$210000$<salt>$<digest>
```

Verification uses `hmac.compare_digest`. Three incorrect attempts end
the command. Set or rotate the PIN with:

```
python scripts/enroll.py set-pin
```

The configured PIN is read without echo. For non-interactive use,
`GREETO_ADMIN_PIN` supplies the current PIN and `GREETO_NEW_PIN`
supplies a new one for `set-pin`. Environment variables are convenient
for automation but are not a substitute for terminal-only entry on a
shared machine.

Keep `config/admin.yaml` private once it contains a hash. The application
does not read it; it is only used by `scripts/enroll.py`.

## 3. Enrolment

```
python scripts/enroll.py enroll --name "Asha Rao" --images photos/asha
python scripts/enroll.py enroll --name "Asha Rao" --images a.jpg b.jpg
python scripts/enroll.py enroll --name "Asha Rao" --images new/ --replace
```

Every image must decode and contain exactly one face. The same quality
gates as recognition apply:

- detection score at least `min_detection_score` (default `0.55`);
- the shorter face side at least `min_face_size` (default `60` px);
- a valid, appropriately sized, finite embedding.

All images are validated before anything is written. A missing face, a
group photo, a tiny face or a corrupt image aborts the whole command and
leaves an existing identity untouched. Without `--replace`, new photos
are appended as the next numbered `NNN.jpg`; with `--replace`, the old
identity folder is swapped out atomically after staging succeeds.

Images are written as decoded pixels, so what the model evaluated is
what the database stores. After writing:

1. the old embedding cache is deleted;
2. the full face database is rescanned and the cache is rebuilt;
3. the command asks for a kiosk restart.

If the cache rebuild fails, the images remain enrolled and the command
prints a warning. The next startup rebuilds from the images.

Other admin commands:

```
python scripts/enroll.py list
python scripts/enroll.py remove "Asha Rao"
python scripts/enroll.py visitors
```

`list` and `visitors` do not load InsightFace. `remove` deletes the
identity folder and invalidates the cache. `visitors` shows the SQLite
statistics and performs the one-time legacy import.

Useful optional paths for testing or a staged deployment:

```
--faces-dir DIR
--cache-file FILE
--admin-config FILE
```

`visitors` also accepts `--database FILE` and `--legacy-json FILE`.
Supplying `--database` defaults the legacy path to a sibling
`visitors.json`, so a temporary database never reads or archives the
real project data.

## 4. Limitations

1. The legacy import intentionally preserves names only; historical
   visit counts are not trustworthy and are not migrated.
2. There is no per-visit UI or reporting command beyond the aggregate
   table printed by `visitors`.
3. SQLite is local to one kiosk process. Multi-process concurrent
   writers were not part of this milestone; WAL and the busy timeout
   provide normal SQLite protection.
4. PIN lockout is per command invocation, not a persistent account
   lockout.
5. Enrolment uses still images only; there is no live-camera capture or
   guided pose flow.
6. A running kiosk must be restarted after enrolment; it does not hot
   reload the face database.
7. `first_seen`/`last_seen` use local time without a timezone offset,
   matching the previous JSON format.

## 5. Tests

```
python tests/test_storage.py
python tests/test_enrollment.py
```

`test_storage.py` covers schema creation, counts, persistence,
case-insensitive names, validation, legacy import, invalid legacy files,
corrupt database recovery, newer-schema refusal and concurrent writes.

`test_enrollment.py` covers name validation, image quality rejection,
all-or-nothing batches, add/replace/remove, cache rebuild and
invalidation, PIN hashing and the CLI authentication gate. It uses
synthetic images and a fake detector; it does not load InsightFace.

The real InsightFace model was additionally exercised once through the
CLI against `models/faces/Saksham/1.jpg` in a temporary face database:
one image was enrolled and one cache entry was rebuilt.
