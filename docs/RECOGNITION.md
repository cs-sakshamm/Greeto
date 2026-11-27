# Face Recognition

Phase 3, Milestone 2. This document covers embedding validation,
quality gates, threshold configuration, structured results and visitor
IDs. Enrolment, the visitor database, greetings and tracking behaviour
are otherwise unchanged.

## Pipeline

```
frame
  └─ InsightFace detect
       └─ normalize + validate embedding       (embedding_model.py)
            └─ quality gate                    (face_recognizer.py)
                 └─ cosine similarity vs known vectors
                      └─ RecognitionResult on Face
                           └─ VisitorTracker assigns visitor_id
                                └─ EventBus: visitor.detected / visitor.unknown
```

Recognition no longer treats every detection as a comparable face.
A face can now be rejected before scoring, and a score below the
threshold is reported as `unknown` rather than silently becoming
`Unknown` with no explanation.

## Embedding rules

- Every embedding is converted to a one-dimensional `float32` vector.
- It must be non-empty, finite, and have the configured
  `expected_embedding_size` (512 for `buffalo_l`).
- Its L2 norm must be greater than `min_embedding_norm` (default `0.5`).
- With `normalize_embeddings: true`, it is divided by its L2 norm.
  Normalisation is scale-invariant: an embedding stored with norm 20.9
  and the same embedding stored at norm 1.0 compare identically.
- A detection whose embedding fails these checks is dropped at
  ingestion. The rest of the frame is unaffected.
- The recogniser validates again when building its known-vector cache
  and when scoring a query, so a corrupt cache cannot inject `nan` or a
  shape mismatch into a comparison.

`cosine_similarity()` is strict and returns a value clamped to
`[-1, 1]`. Mismatched sizes, zero vectors, empty vectors and non-finite
values raise `ValueError`; callers turn that into a structured status.

## Recognition configuration

`config/vision.yaml`:

```yaml
recognition:
  threshold: 0.55
  min_detection_score: 0.55
  min_face_size: 60
  expected_embedding_size: 512
  min_embedding_norm: 0.5
  normalize_embeddings: true
```

| Key | Default | Meaning |
| --- | --- | --- |
| `threshold` | `0.55` | Minimum cosine similarity for a match |
| `min_detection_score` | `0.55` | InsightFace detector score gate |
| `min_face_size` | `60` | Minimum face side in pixels |
| `expected_embedding_size` | `512` | Required embedding length |
| `min_embedding_norm` | `0.5` | Reject near-zero vectors |
| `normalize_embeddings` | `true` | L2-normalise before comparison |

Invalid values do not stop startup: each one is replaced by its default
and logged as `Recognition config: ...`. Missing keys use defaults too.
`--threshold` is not a runtime CLI option; change the YAML and restart.

The twice-checked threshold is the only decision boundary. There is no
per-identity threshold in this milestone.

## Structured result

`Face.recognition` is a `RecognitionResult`:

| Field | Meaning |
| --- | --- |
| `status` | One of the statuses below |
| `name` | Matched identity, or `"Unknown"` |
| `recognized` | `True` only for `recognized` |
| `score` | Best cosine similarity, or `None` if never compared |
| `threshold` | Threshold used for this attempt |
| `candidate` | Closest identity even when below threshold |
| `reason` | Human-readable rejection/decision detail |

Statuses:

| Status | Meaning | `score` |
| --- | --- | --- |
| `not_processed` | No recognition attempted yet | `None` |
| `recognized` | Best score at or above threshold | similarity |
| `unknown` | Compared, but best score below threshold | similarity |
| `low_quality` | Failed detection-score or face-size gate | `None` |
| `invalid_embedding` | Missing/malformed/non-finite embedding | `None` |
| `no_known_faces` | Database empty, unavailable, or nothing comparable | `None` |

`Face.name`, `Face.recognized` and `Face.confidence` remain compatible
with existing code. `Face.confidence` is always the **detector** score;
similarity is exposed through `Face.recognition_score` and never
overwrites detection confidence. `Face.recognition_status` defaults to
`not_processed` when no result exists.

Quality gates run in this order and never compare embeddings:

1. usable bounding box (`width > 0`, `height > 0`);
2. finite detection score;
3. detection score at least `min_detection_score`;
4. `min(width, height)` at least `min_face_size`.

## Threshold calibration

`scripts/calibrate_threshold.py` loads the face database and computes:

- **genuine** scores: all pairs of photos of the same person;
- **impostor** scores: cross-identity pairs, capped by `--max-pairs`.

`calibrate_threshold()` searches a grid (`--low`, `--high`, `--steps`)
for maximum balanced accuracy, breaking ties towards the widest gap
between the highest impostor and lowest genuine score. It returns
FAR, FRR and the counts used.

Without both pair types the script refuses to recommend a number and
keeps the configured threshold. It is read-only; it prints the YAML
change rather than applying it.

Current database (one identity, one photo):

```
identities        : 1
embeddings        : 1
genuine pairs     : 0
impostor pairs    : 0
Not enough data to calibrate safely.
Keeping the configured threshold of 0.550.
```

This is expected: one photo per person cannot produce genuine pairs,
and one person cannot produce impostors. Add several photos per person
and at least two people, then rerun. Calibration on enrolled still
photos is an estimate, not a substitute for checking live-camera
scores.

## Visitor IDs

`VisitorTracker` assigns IDs with centroid distance and a maximum
matching distance. Milestone 2 fixes a real collision: previously two
overlapping faces in the same frame could both match the same track and
receive the same `visitor_id`. Matched tracks are now excluded within a
frame, so every face gets a distinct ID.

- IDs increase monotonically for the life of the tracker; an expired ID
  is not reused by a later person.
- `clear()` starts a new session and resets numbering; this is the only
  case where numbering restarts.
- Movement within `max_distance` keeps the same ID.
- Missing tracks expire after `max_missing_frames`; the default remains
  30 frames.

## Events

Existing event names are unchanged. `visitor.detected` and
`visitor.unknown` now also carry:

| Field | Value |
| --- | --- |
| `visitor_id` | Track ID for this visitor |
| `score` | Recognition similarity, or `None` if not compared |
| `status` | The structured status string |
| `face` | The `Face` object (as before) |

`visitor.detected` additionally has `name`. The kiosk, greeting timer
and TTS paths are untouched; subscribers that accept `**payload` keep
working. Each visitor is announced once per tracking session.

## Diagnostics

- Startup logs: `Recognition ready: N identities, M embeddings
  (threshold X, R rejected).`
- `VisionService.health()["recognition"]` returns threshold, known
  identity/embedding counts, rejected embeddings, and the last status,
  score and candidate.
- `F10` in the kiosk shows the threshold, known counts and last status
  alongside camera health.
- Drawn labels use the similarity score when the face was compared, and
  fall back to the detector score for a quality-rejected face.

## Cache handling

`models/cache/face_embeddings.pkl` is derived data.

- An unreadable cache is logged and rebuilt from `models/faces`.
- A cache with no usable embeddings is rebuilt.
- Partly invalid entries are dropped, the usable ones are loaded, and
  the cleaned mapping is saved back.
- A cache may contain raw embeddings of any valid norm; the recogniser
  normalises its in-memory copy and does not rewrite the cache just for
  scale.
- If rebuilding finds no usable faces, recognition reports
  `no_known_faces`; it does not crash the camera pipeline.

## Tests

`tests/test_recognition.py` uses numpy and in-memory fakes only: no
camera, no InsightFace model, no GUI. It covers validation,
normalisation, cosine edge cases, every status, quality gates, config
fallbacks, threshold decisions, confidence preservation, stats,
calibration, cache sanitation, unique IDs, movement/expiry, structured
events and health output.

Actual result:

```
python tests/test_recognition.py
RESULT: ALL CHECKS PASSED   (21/21)
```

Real-image sanity check (`models/faces/Saksham/1.jpg`, real
InsightFace, real cache): detection score `0.904`, unit-norm embedding,
`recognized Saksham`, similarity `0.99995`.

## Limitations

1. The threshold remains at the existing `0.55`; it has not been
   data-calibrated because there are no genuine or impostor pairs yet.
2. There is still only one enrolled identity. Cross-person false accepts
   cannot be measured until more people are enrolled.
3. Quality gates are fixed heuristics (detection score and face size),
   not learned quality assessment. Pose, blur and illumination are not
   measured separately.
4. Recognition remains single-frame and per-face; no temporal voting or
   embedding averaging across frames is done.
5. The cache stores raw-norm vectors; normalisation happens in memory
   at load. This is intentional and does not affect scores.
6. Live-camera score distributions were not collected; the real-image
   check only proves the pipeline works on an enrolled photo.
