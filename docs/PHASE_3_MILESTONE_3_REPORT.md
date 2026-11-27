# Phase 3 — Milestone 3 Implementation Report

**Scope:** persistent visitor storage and admin enrolment.
**Status:** complete; targeted, regression and real-model checks pass.
**Date:** 23 November 2026.

## 1. What changed

### Files modified

| File | Change |
| --- | --- |
| `app/vision/visitor_database.py` | Replaced JSON rewriting with a transactional SQLite store, lazy initialisation, legacy import, corruption recovery and visit history |
| `app/vision/enrollment.py` | Filled the empty placeholder with validated, all-or-nothing image enrolment, replacement, removal and cache rebuild |
| `app/vision/vision_service.py` | Reads visitor storage paths from config, accepts an injected store for tests, records visits without letting storage errors stop greeting, exposes visitor stats in health |
| `config/vision.yaml` | Added the `visitors` section for the database and legacy-file paths |
| `tests/test_recognition.py` | Injects the visitor store; health check now covers visitor statistics |
| `.gitignore` | Ignores runtime SQLite files and one-time legacy archives |

### Files created

| File | Purpose |
| --- | --- |
| `app/core/admin_auth.py` | PIN validation, salted PBKDF2 hashing, constant-time verification and admin YAML storage |
| `config/admin.yaml` | Empty admin-PIN template; contains a hash only after `set-pin` |
| `scripts/enroll.py` | PIN-gated CLI for set-pin, enrol, list, remove and visitors |
| `tests/test_storage.py` | 10 hardware-free SQLite, migration, recovery and concurrency checks |
| `tests/test_enrollment.py` | 12 enrolment, PIN and CLI checks with synthetic images |
| `docs/STORAGE.md` | Storage, migration, enrolment and PIN documentation |
| `docs/PHASE_3_MILESTONE_3_REPORT.md` | This report |

No files were deleted or moved. No dependency was added: SQLite,
PBKDF2 and YAML support were already available.

## 2. Decisions

1. **SQLite with two tables, no duplicated counter.** `visitors` holds
   identity and first/last timestamps; `visits` holds one row per
   announced visit. The displayed visit count is queried from `visits`,
   so it cannot drift from the rows.
2. **Record one visit per announcement.** The existing tracking and
   announcement gate is kept. The kiosk does not write on every frame,
   and unknown faces are not stored.
3. **Import legacy identities but discard legacy statistics.** The old
   `visits` values were incremented on every detection cycle and are not
   traffic data. Names are imported once; timestamps and counters are
   discarded. Imported records show zero visits until the next real
   visit.
4. **Archive, do not silently delete.** A successful import moves
   `visitors.json` to `visitors.json.migrated`; a malformed file becomes
   `visitors.json.invalid*`. The original content remains available for
   inspection. The database is never imported twice.
5. **Lazy initialisation.** Constructing `VisitorDatabase` has no
   filesystem side effects, so importing the vision stack or running
   tests does not create a real database. Schema creation and migration
   happen on the first read or write.
6. **Recover from a corrupt database.** A non-SQLite file is archived as
   `visitors.db.corrupt-<timestamp>` and replaced. A database written by
   a newer application schema is refused instead of being modified.
7. **Storage failure must not stop reception.** `VisionService` logs a
   failed visit write and still publishes the greeting event.
8. **No plaintext PIN.** The PIN is 4–12 digits, salted with 16 random
   bytes and hashed with PBKDF2-HMAC-SHA256 at 210,000 iterations.
   Verification is constant-time. Three wrong attempts end the command.
9. **CLI-only administration was in scope.** There is no kiosk button or
   on-screen admin flow; enrolment stays an offline maintenance task.
10. **Validate the whole batch before writing.** Every image must decode
    and contain exactly one face that passes the recognition quality
    gates. A bad image aborts the command before the identity folder or
    cache is changed.
11. **Re-encode decoded pixels.** The stored image is what the detector
    actually evaluated, rather than an arbitrary source file with
    unexpected metadata.
12. **Invalidate and rebuild the embedding cache.** Enrolment deletes
    the old cache, rescans the full face folder and writes a fresh
    cache. Removal deletes the identity and invalidates the cache.
13. **Leave the old files in place until first use.** The checked-in
    `data/visitors.json` was not moved during testing. The first real
    visitor or `visitors` command performs the import and archives it.
14. **No scope expansion.** Greeting logic, recognition thresholds,
    tracking, speech, the LLM, TTS and the face-folder schema were not
    changed.

## 3. Tests executed and actual results

Environment: Windows, `.venv` Python 3.11.1; suites run serially.

| Command | Result |
| --- | --- |
| `python tests/test_storage.py` | **ALL CHECKS PASSED** (10/10) |
| `python tests/test_enrollment.py` | **ALL CHECKS PASSED** (12/12) |
| `python tests/test_recognition.py` | **ALL CHECKS PASSED** (21/21) |
| `python tests/test_camera.py` | **ALL CHECKS PASSED** (9 passed; live skipped) |
| `python tests/test_face.py --skip-slow` | **ALL CHECKS PASSED** (10 passed; long run skipped) |
| `python tests/test_kiosk.py` | **ALL CHECKS PASSED** (13 passed; continuous run skipped) |

### Storage checks

| Area | Checks |
| --- | --- |
| Schema/lifecycle | Lazy creation, empty stats, first read, visit creation |
| Counting | Repeated visits, distinct visitors, first/last timestamps, totals |
| Persistence | New instance reads the same records; display name is preserved |
| Names | Whitespace collapse, case-insensitive lookup, invalid values rejected |
| Legacy import | Names imported, old counts/timestamps discarded, archive created, no second import |
| Invalid legacy file | Archived, no import, database still usable |
| Corrupt database | Archived, fresh database works |
| Schema safety | Newer schema refused |
| Isolation | Temporary database uses only temporary paths |
| Concurrency | 40 writes from 40 threads, no lost visits |

### Enrolment/admin checks

| Area | Checks |
| --- | --- |
| Names | Valid Unicode/space/apostrophe names; traversal, control and Windows-reserved names rejected |
| Enrolment | Two images written, numbering, cache rebuilt with two embeddings |
| Duplicate sources | Same path supplied twice is enrolled once |
| Add/replace | Append gives 001/002; replace leaves exactly one file |
| Rejection | No face, two faces, low detection score and tiny face each leave no identity and no cache |
| Failed replacement | Existing identity is unchanged |
| Bad sources | Missing path, undecodable image and empty directory are rejected |
| List/remove | Empty folders ignored; removal deletes folder and invalidates cache |
| PIN | Correct/incorrect PIN, empty/malformed hash, different salts, tampered hash, invalid PIN lengths |
| PIN config | Hash round-trip, plaintext never present, invalid hash refused |
| CLI | `set-pin`, wrong-PIN gate before image inspection, `list`, `visitors` |

### Real-model check

The real InsightFace `buffalo_l` model was run once through the CLI
against the enrolled `models/faces/Saksham/1.jpg`, with a temporary face
folder and temporary cache:

```
[FaceDatabase] Loading Saksham Test...
  Loaded 001.jpg
[FaceDatabase] Cache saved.
[FaceDatabase] Loaded 1 people.
[FaceDatabase] Total embeddings: 1
Enrolled Saksham Test: added 1 image(s), 1 total.
Embedding cache rebuilt.
```

The temporary cache contained exactly one identity. The project face
folder and cache were not modified.

### Legacy migration check

The real `data/visitors.json` was copied to a temporary directory and
imported through the CLI:

```
Visitors: 1    recorded visits: 0
Legacy import: 1 identities imported; old visit counters were discarded.
Legacy file archived to: ...\visitors.json.migrated

Name                           Visits  First seen           Last seen
Saksham                             0  -                    -
```

That is the intended contract: identity preserved, unreliable `1162`
visit count dropped.

## 4. Existing functionality preserved

- Camera, face animation and kiosk suites pass unchanged.
- Recognition still uses the same thresholds, quality gates, event names
  and `visitor_id`/`score`/`status` fields.
- Greeting and duplicate protection still record one visit per tracked
  visitor announcement, not one per frame.
- Unknown faces still trigger no visitor-storage write.
- Speech, Whisper, Piper, the local LLM and conversation handling were
  not modified.
- `models/faces`, `models/cache/face_embeddings.pkl` and
  `data/visitors.json` were not changed by this milestone.
- No second visitor database, tracking path or greeting path was added.
- The stale `app/test_*.py` scripts remain untouched for the cleanup
  milestone.

## 5. Known limitations and unresolved risks

1. **Legacy statistics are intentionally not migrated.** Imported names
   have zero visits and no timestamps until their next real visit.
2. **The real `visitors.json` is still present.** The migration runs on
   the first visitor or `visitors` command. This avoids changing live
   data during the implementation, but means the first run after
   deployment performs the archive step.
3. **No per-visit reporting UI.** `visitors` prints aggregate rows; there
   is no query command for individual visit times.
4. **PIN lockout is per invocation.** There is no persistent lockout or
   audit trail across runs. The environment-variable path is intended
   for automation, not a hardened multi-user deployment.
5. **Enrolment uses still images only.** There is no live-camera capture,
   pose guidance or automatic multi-shot session.
6. **The kiosk must be restarted after enrolment.** The running process
   does not hot reload the face database.
7. **Multi-process SQLite writers were not tested.** WAL and the busy
   timeout provide normal SQLite safety, but a single kiosk process is
   the assumed deployment.
8. **Live camera, long stability and multi-identity calibration checks
   were not run.** The opt-in camera and stability tests were skipped.
9. **Only one real identity/one real photo exists.** Enrolment quality
   and cache rebuild were verified with that photo; cross-person
   enrolment accuracy still needs more people.

## 6. Recommendation and approval request

Milestone 3 meets its scope: SQLite replaced the JSON store, the legacy
visit counters are not carried forward, admin enrolment is PIN-gated and
validated, and the existing recognition, camera and kiosk behaviour is
preserved. All checks listed above were actually run and passed; skipped
opt-in runs are identified as skipped, not passed.

I recommend the next milestone be the remaining approved cleanup and
speech hardening: a shared `TTSService`, review of the stale
`app/test_*.py` scripts, and any remaining repository cleanup. Please
confirm the exact scope for **Milestone 4** before I begin.

**Approval requested:** confirm Milestone 4. I will stop here until you
approve.
