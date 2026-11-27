# Phase 3 — Milestone 2 Implementation Report

**Scope:** recognition engine and embedding reliability.
**Status:** complete; targeted and regression suites pass.
**Date:** 23 November 2026.

## 1. What changed

### Files modified

| File | Change |
| --- | --- |
| `app/vision/models.py` | Added `RecognitionResult`, status constants, `Face.recognition`, `recognition_score`/`recognition_status`; detection confidence preserved |
| `app/vision/similarity.py` | Replaced permissive helpers with strict embedding validation/normalisation and clamped cosine similarity |
| `app/vision/embedding_model.py` | Validates and L2-normalises embeddings; drops detections with malformed vectors |
| `app/vision/face_database.py` | `sanitize_embeddings()`, corrupt/empty-cache rebuild, partial-cache rewrite, optional cache injection for tests |
| `app/vision/face_recognizer.py` | Config injection, normalised known-vector cache, quality gates, structured results, stats |
| `app/vision/visitor_tracker.py` | Per-frame track exclusion so overlapping faces cannot share an ID; monotonic IDs |
| `app/vision/vision_service.py` | Loads recognition config, refreshes recogniser after database load, logs readiness, exposes recognition health, adds structured fields to events |
| `app/gui/main_window.py` | F10 diagnostics show recognition threshold, known counts and last status |
| `config/vision.yaml` | Filled the previously empty file with documented recognition settings |

### Files created

| File | Purpose |
| --- | --- |
| `app/vision/recognition.py` | `RecognitionConfig`, defaults/fallback parsing, `calibrate_threshold()` |
| `tests/test_recognition.py` | 21 hardware-free checks; no camera, model or GUI |
| `scripts/calibrate_threshold.py` | Read-only genuine/impostor score calibration report |
| `docs/RECOGNITION.md` | Recognition contract, configuration, events, calibration and limitations |
| `docs/PHASE_3_MILESTONE_2_REPORT.md` | This report |

No files were deleted or moved. No dependency was added. No data,
model or visitor-database schema was changed.

## 2. Decisions

1. **Validate at both ends.** Embeddings are normalised and validated
   when InsightFace produces them, and again when the recogniser builds
   its known-vector cache and scores a query. A bad cache entry or a
   hand-built `Face` therefore cannot reach a comparison.
2. **Keep detector confidence separate.** `Face.confidence` remains the
   InsightFace detection score. Similarity lives in
   `Face.recognition.score`; overwriting confidence would have hidden
   detection quality and broken existing consumers.
3. **Fail softly per face, not per frame.** Every attempt produces a
   `RecognitionResult`; one rejected face never aborts the batch or the
   camera pipeline.
4. **Quality gates before comparison.** Detection score below `0.55` or
   a face side below `60 px` yields `low_quality` with no similarity
   score. These are the existing useful thresholds, now explicit.
5. **Threshold remains `0.55`.** This was the working default and the
   only value supported by real evidence (an enrolled photo scoring
   `0.99995`). Calibration is available but is not applied
   automatically; with one identity and one photo there is nothing to
   calibrate.
6. **No automatic calibration.** The calibration function is pure and
   testable; the CLI is read-only. A data-derived threshold should be
   reviewed and set in YAML, not silently changed at startup.
7. **Additive event fields.** `visitor.detected` and `visitor.unknown`
   gain `visitor_id`, `score` and `status`; event names and the `face`
   payload are unchanged, so existing subscribers keep working.
8. **Fix the tracker collision rather than masking it.** The old
   matching loop could reuse one track for two faces in a frame. Matched
   tracks are now excluded in-frame; IDs increase monotonically and are
   not reused after expiry.
9. **Treat the cache as derived data.** Corrupt, empty or partly invalid
   caches rebuild or self-clean instead of preventing recognition.
10. **No scope expansion.** SQLite migration, enrolment, greeting
    changes, new tracking algorithms and `data/visitors.json` changes
    were deliberately left for later milestones.

## 3. Tests executed and actual results

Environment: Windows, `.venv` Python 3.11.1; suites run serially.

| Command | Result |
| --- | --- |
| `python tests/test_recognition.py` | **ALL CHECKS PASSED** (21/21) |
| `python tests/test_camera.py` | **ALL CHECKS PASSED** (9/9; live skipped) |
| `python tests/test_face.py --skip-slow` | **ALL CHECKS PASSED** (10/10; stability skipped) |
| `python tests/test_kiosk.py` | **ALL CHECKS PASSED** (13/13; continuous run skipped) |
| `python scripts/calibrate_threshold.py` | Exit 0; correctly reported insufficient data |

Recognition checks:

| Area | Checks |
| --- | --- |
| Embeddings | unit length, float32, scale invariance, wrong size/shape/zero/NaN/non-numeric rejection |
| Cosine | identical, orthogonal, opposite, scaling, bounded result, size/zero/infinite guards |
| Config | defaults, explicit overrides, numeric strings, invalid values fall back with warnings, threshold validation |
| Results | recognized, unknown with candidate, low quality (score/size/box/NaN), invalid embedding, empty/broken database |
| Batch | mixed good/bad faces all processed; one bad face does not abort |
| Threshold | same score accepted at `0.70`, rejected at `0.90` |
| Stats | threshold, known identities/embeddings, rejected count, last status/score |
| Visitor IDs | two and fifty overlapping faces get distinct IDs; stability; expiry; movement; `clear()` |
| Calibration | clear separation, overlap, missing impostors, NaN filtering, invalid range |
| Cache | sanitation, unreadable cache, valid cache, partial rewrite, empty fallback |
| Integration | structured events carry `visitor_id`/`score`/`status`; no duplicate announcements; health includes recognition |

Real-image sanity check (real InsightFace, real cache, no writes):

```
refresh: (1, 1, 0)
detected: 1
result: Saksham True recognized 0.99995 det 0.904
stats: threshold 0.55, known_identities 1, known_embeddings 1,
       last_status recognized, last_candidate Saksham
```

Calibration output on the current database:

```
identities        : 1
embeddings        : 1
genuine pairs     : 0
impostor pairs    : 0

Not enough data to calibrate safely.
- need at least two photos of the same person (currently one photo per identity)
- need at least two enrolled people

Keeping the configured threshold of 0.550.
```

No live-camera score distribution was collected in this milestone.

## 4. Existing functionality preserved

- Kiosk greeting, duplicate-greeting protection, stale-TTS handling,
  microphone-unavailable handling, cross-thread events and clean
  shutdown all pass (`tests/test_kiosk.py`, 13/13).
- Face animation, states, transitions, resizing and MainWindow
  integration pass (`tests/test_face.py`, 10/10).
- Camera health, reconnect, degraded startup and shutdown pass
  (`tests/test_camera.py`, 9/9).
- Speech, Whisper, Piper, the local LLM and `VisitorDatabase` behaviour
  were not modified.
- `data/visitors.json` and its schema were not touched.
- The stale `app/test_*.py` scripts were not used or changed.
- Existing event names and payload keys remain; recognition fields are
  additive.
- No new dependency; only numpy, InsightFace, OpenCV and existing
  project modules are used.

## 5. Known limitations and unresolved risks

1. **Threshold is not data-calibrated.** With one identity and one photo
   there are no genuine or impostor pairs. `0.55` remains the configured
   default.
2. **Single identity.** Cross-person false accepts cannot be measured
   until at least two people are enrolled with multiple photos.
3. **Calibration uses still photos.** Even when enough data exists, the
   result estimates pair separability, not live-camera behaviour.
4. **Quality gates are heuristics.** Detection score and face size do
   not directly measure blur, pose or illumination.
5. **No temporal aggregation.** Recognition is per frame; no voting or
   embedding averaging across frames.
6. **Real-image check is one enrolled photo.** It proves the model,
   cache, normalisation and matching path work, not general accuracy.
7. **Live camera recognition was not exercised** in this milestone;
   camera capture and recognition were verified separately.
8. **F10 diagnostics remain development-only.** Recognition health is
   not shown to visitors.

## 6. Recommendation and approval request

Milestone 2 meets its stated scope: embeddings are validated and
normalised, quality gates prevent bad comparisons, the threshold is
configurable, results are structured, and visitor IDs are unique per
frame. All targeted checks and all regression suites actually run here
pass.

I recommend proceeding to **Milestone 3: persistent visitor storage and
enrolment** — the previously approved SQLite migration, removal of the
legacy `visitors.json` statistics, and CLI + PIN admin enrolment, while
keeping the recogniser and kiosk behaviour intact.

**Approval requested:** please confirm Milestone 3. I will stop here
until you approve.
