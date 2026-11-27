# Greeto — Phase 3 Milestone 4 Audit Report

## 1. Original Milestone Scope

**Phase 3 – Milestone 4: Visitor Tracking and Session Intelligence**

Its intended scope included:

1. Persistent visitor identity association.
2. Reliable face-to-track association across frames.
3. Arrival, presence, and departure events.
4. Grace periods for temporary disappearance.
5. Return recognition within sessions.
6. Duplicate-event suppression.
7. Support for multiple independent visitor sessions.
8. Integration with the existing event system.

---

## 2. Actual Repository State at Audit Beginning

### Git Status

- Branch: `test` (root commit `ad0f3cd`)
- No prior commits beyond the initial one
- Modified files (28 files): `.gitignore`, `app/brain/conversation_manager.py`, `app/core/config_manager.py`, `app/core/container.py`, `app/core/event_bus.py`, `app/core/logging_setup.py`, `app/gui/camera_widget.py`, `app/gui/main_window.py`, `app/gui/robot_widget.py`, `app/gui/theme.py`, `app/main.py`, `app/speech/audio_capture.py`, `app/speech/speech_service.py`, `app/speech/stt.py`, `app/speech/tts_service.py`, `app/vision/camera_manager.py`, `app/vision/embedding_model.py`, `app/vision/enrollment.py`, `app/vision/face_cache.py`, `app/vision/face_database.py`, `app/vision/face_recognizer.py`, `app/vision/frame_grabber.py`, `app/vision/models.py`, `app/vision/similarity.py`, `app/vision/vision_service.py`, `app/vision/visitor_database.py`, `app/vision/visitor_tracker.py`, `config/camera.yaml`, `config/vision.yaml`
- Deleted files (5): `app/test_database.py`, `app/test_face_recognizer.py`, `app/test_speech.py`, `app/test_vision.py`, `data/visitors.json`
- Untracked files (18): `app/core/admin_auth.py`, `app/gui/caption_widget.py`, `app/gui/face_animation.py`, `app/gui/face_renderer.py`, `app/gui/kiosk_background.py`, `app/gui/kiosk_presenter.py`, `app/gui/resources/robot.svg`, `app/gui/vision_pump.py`, `app/vision/recognition.py`, `config/admin.yaml`, `docs/`, `requirements_jetson.txt`, `requirements_windows.txt`, `scripts/`, `tests/`

### Key Implementation Files Inspected

| File | Purpose | Key Changes |
|------|---------|-------------|
| `app/speech/tts_service.py` | Piper TTS wrapper with lock-based serialization | Single shared instance, `threading.Lock()` covering full generate-and-play cycle |
| `app/brain/conversation_manager.py` | AI conversation controller | Receives `tts_service` via DI, no internal TTS construction |
| `app/main.py` | Application entry point | Creates one `TTSService`, injects into `ConversationManager` |
| `app/vision/visitor_tracker.py` | Frame-to-frame visitor tracking | Centroid-based tracking, stable IDs, expiration/cleanup |
| `app/vision/visitor_database.py` | SQLite visitor persistence | Lazy schema, legacy JSON migration, concurrent-safe |
| `app/core/event_bus.py` | Thread-safe publish/subscribe | Failing subscribers don't abort bus or background thread |

---

## 3. Discrepancy Register

| Reported Claim | Actual Repository Evidence | Status | Required Action |
|---|---|---|---|
| **Single TTS instance** | `main.py` creates exactly one `TTSService(container)` and injects it as `tts_service=tts` into `ConversationManager(container, tts_service=tts)`. No other constructor creates a TTS instance. | **Verified** | None |
| **Speech serialization** | `TTSService._speak()` runs `with self._speak_lock:`, covering utterance numbering, Piper subprocess, playback, event publishing, and cleanup. Tests pass with 2 concurrent threads -> one at a time, one WAV each. | **Verified** | None |
| **Event ordering** | `speech.speaking_started` published after Piper WAV generated and immediately before playback. `speech.speaking_finished` published in `finally` block after WAV deletion. Events carry `utterance` number. Test asserts `{started_utterances} == {finished_utterances}`. | **Verified** | None |
| **Failure cleanup** | Failed Piper (exit code != 0) -> `speech.error` event, no playback, temporary WAV deleted. Exceptions in `_speak` -> `speech.error` event, `finally` block ensures WAV cleanup and `speaking_finished` only if `started=True`. | **Verified** | None |
| **Deleted scripts were unused** | `app/test_database.py` - simple starter script, equivalent behavior in `tests/test_storage.py`. `app/test_face_recognizer.py` - face recog test, equivalent in `tests/test_recognition.py`. `app/test_speech.py` - SpeechService starter, equivalent in `tests/test_speech.py`. `app/test_vision.py` - VisionService starter, equivalent in `tests/test_recognition.py`/`tests/test_face.py`. `data/visitors.json` - legacy JSON properly migrated to SQLite with visit counters discarded. | **Verified** | None |
| **All tests passed** | `test_speech.py`: ALL CHECKS PASSED (4/4). `test_storage.py`: ALL CHECKS PASSED (23/23). `test_enrollment.py`: ALL CHECKS PASSED (34/34). `test_recognition.py`: ALL CHECKS PASSED (35/35). | **Verified** | None |
| **Phase 3 Milestone 4 complete** | **NOT VERIFIED** - The actual implementation covers TTS integration and repository cleanup, NOT the originally planned Visitor Tracking and Session Intelligence scope. See Section 7 for detailed gap analysis. | **Incorrect** | Document distinction; assess visitor tracking completeness separately |
| **TTS ownership is unambiguous** | One `TTSService` created in `main()`, injected into `ConversationManager` via DI. No private TTS construction inside `ConversationManager`. `manager.tts is tts` assertion passes. | **Verified** | None |
| **Concurrent speech is safe under tested conditions** | Test spawns 2 threads calling `service.speak()` simultaneously. Lock ensures one-at-a-time execution. `player.max_active == 1` assertion passes. No overlapping playback. | **Verified** | None |
| **Piper failure and cleanup paths verified** | Test simulates Piper exit code 3. Asserts: `result is False`, `player.played == []`, `events == [speech.error]`, `created[0].output_path` deleted. Empty text -> `service.speak("") is False`, no events published. | **Verified** | None |
| **Speech event behavior is correct** | `speech.speaking_started` payload: `text`, `utterance`. `speech.speaking_finished` payload: `utterance`. `speech.error` payload: `kind`, `message`. Order: lock -> `_utterance += 1` -> Piper -> `_publish(speaking_started)` -> playback -> `_publish(speaking_finished)` in `finally`. Failures emit `error`, not `speaking_finished`. | **Verified** | None |
| **Phase 3 Milestone 3 integration intact** | SQLite visitor storage works: lazy initialization, legacy JSON migration with visit counters discarded, corrupt database archival, schema versioning, concurrent registration loses no visits. All 23 storage tests pass. | **Verified** | None |
| **Visitor tracking implemented** | `VisitorTracker` class implements centroid-based tracking with stable IDs. Tests pass: overlapping faces get distinct IDs (`[1, 2]`), expired IDs not reused within session, centroid movement keeps same ID, 50 faces -> 50 unique IDs. However, higher-level session intelligence (arrival/departure events, grace periods, multiple independent sessions) is NOT implemented. | **Partial** | See Section 7 |
| **No real visitor data changed** | All tests use temporary directories. No modification to real visitor database, enrollment images, or face embeddings. | **Verified** | None |

---

## 4. TTS Integration Verification

### B1. Single-instance Ownership ✅ VERIFIED

- `main.py` line: `tts = TTSService(container)` — exactly one constructor call
- `main.py` line: `ConversationManager(container, tts_service=tts)` — injected, not constructed internally
- `conversation_manager.py` line: `self.tts = tts_service` — stores injected instance
- No hidden constructors or fallback paths found
- Service remains alive for application lifetime (created before window, stopped in `finally` block)

### B2. ConversationManager Dependency Injection ✅ VERIFIED

- `tts_service` is explicitly required as constructor parameter
- No private TTS instance constructed inside `ConversationManager` (`self.tts = tts_service`)
- Existing conversation handling still works (tests pass)
- Parsed answer passed to TTS: `self.tts.speak(answer)`
- Missing dependency would fail clearly (TypeError if `tts_service` not provided)
- Tests use injected stub: `test_conversation_manager_uses_the_shared_service` passes

### B3. Thread Safety and Locking ✅ VERIFIED

- `self._speak_lock = threading.Lock()` in `TTSService.__init__`
- Lock acquired in `speak()` -> `_speak()` path
- `_utterance += 1` under lock before Piper launch
- Playback `_play_audio()` called while lock still held
- `speech.speaking_started` published while lock held
- `speech.speaking_finished` published in `finally` while lock held (if `started=True`)
- Exceptions in `_speak` -> `speech.error` in `except` -> `finally` ensures WAV cleanup + `speaking_finished` if started
- Empty text returns `False` before lock acquired
- Test with 2 concurrent threads: `player.max_active == 1`, no overlapping playback
- No deadlock from event subscriber calling `speak()` again (event bus subscribers called outside lock, but `speak()` would block on lock; no recursive re-entry detected in code, but test passes without deadlock)

### B4. Piper Subprocess Lifecycle ✅ VERIFIED

- Piper launched with `subprocess.Popen([\"piper\", \"--model\", ..., \"--output_file\", wav_path], stdin=subprocess.PIPE, text=True)`
- Input text passed via `process.communicate(text)`
- Exit code checked: `if process.returncode not in (0, None): raise RuntimeError`
- WAV file played via platform-appropriate player (`winsound.PlaySound` on Windows, `aplay`/`paplay`/`ffplay` on other)
- Temporary WAV deleted in `finally` block: `Path(wav_path).unlink()` (OSError caught silently)
- Failed process: WAV left behind intentionally for investigation, but `speech.error` emitted and no playback attempted
- No process handles left running unnecessarily (process is short-lived, `communicate()` blocks until complete)

### B5. Speech Event Correctness ✅ VERIFIED

- `speech.speaking_started`: published after WAV generated, before playback. Payload: `text`, `utterance`.
- `speech.speaking_finished`: published in `finally` block. Payload: `utterance`. Means "speaking operation ended for any reason".
- `speech.error`: published on any exception. Payload: `kind="tts"`, `message`.
- Event ordering verified by test: started events == finished events (same utterance numbers).
- Kiosk subscribers receive expected events via event bus.

---

## 5. Repository Cleanup Verification

### Deleted Files Check

| File | Previous Contents (from Git) | References Searched | Decision |
|------|------------------------------|---------------------|----------|
| `app/test_database.py` | Simple script that loaded embedding model and listed people. | `tests/test_storage.py` covers equivalent SQLite behavior. No external workflow depends on it. | **Appropriate** — equivalent tests exist |
| `app/test_face_recognizer.py` | Loaded EmbedlingModel, created FaceRecognizer, recognized image. | `tests/test_recognition.py` covers equivalent recognition behavior. No external workflow depends on it. | **Appropriate** — equivalent tests exist |
| `app/test_speech.py` | Simple starter that created SpeechService and slept on KeyboardInterrupt. | `tests/test_speech.py` covers complete TTS service behavior. No external workflow depends on it. | **Appropriate** — equivalent tests exist |
| `app/test_vision.py` | Created ServiceContainer, VisionService, started/stopped. | `tests/test_recognition.py`/`tests/test_face.py` cover equivalent vision behavior. No external workflow depends on it. | **Appropriate** — equivalent tests exist |
| `data/visitors.json` | `{"Saksham": {"first_seen": "2026-07-30T23:32:43.649948", "visits": 1036, "last_seen": "2026-07-31T01:57:39.713558"}}` | Migrated to SQLite: visit counters discarded (were artefacts), identities imported, file archived as `visitors.json.migrated`. | **Appropriate** — migration preserves data, improves storage |

### `.gitignore` Check

- Original: omitted `Thumbs.db`, no Python bytecode or runtime artefact exclusions
- Current: adds `*.pyo`, `.pytest_cache/`, `build/`, `dist/`, `*.egg-info/`, `.env`, `data/*.db`, `data/*.db-*`, `data/*.json.migrated`, `data/*.json.invalid`, `.idea/`, `logs/`, `.face_tmp/`, `.stability.log`
- No source files accidentally hidden
- Runtime database files (`data/*.db`) now properly ignored — appropriate for a deployed kiosk

---

## 6. Phase 3 Milestone 3 Integration ✅ VERIFIED

All Milestone 3 work (SQLite visitor storage and enrollment) remains intact:

- **Visitor database initialization**: `VisitorDatabase.initialize()` creates schema, runs one-time legacy import
- **Legacy migration behavior**: JSON identities imported without visit counters (correctly treated as artefacts), file archived as `.migrated`
- **Enrollment**: Works correctly — writes images, builds face cache with embeddings, supports add/replace
- **PIN authentication**: `hash_pin`, `verify_pin`, `save_pin_hash`, `load_pin_hash` all functional; CLI enrollment gated by correct PIN
- **Embedding cache invalidation**: `FaceDatabase.load()` with corrupt/empty cache falls back safely
- **Visitor logging**: `VisitorDatabase.register()` records one visit per arrival; `stats()` reports total visitors and visits
- **VisionService integration**: Visitor tracking integrates with face recognition pipeline
- **Recognition event delivery**: Structured events carry `visitor_id`, `name`, `status`, `score`

All 23 storage tests pass, all 34 enrollment tests pass, all 35 recognition tests pass.

---

## 7. Visitor Tracking and Session Intelligence — Implementation Gap Report

### Requirement Status

| Requirement | Evidence | Status |
|---|---|---|
| **Face-to-track association** | `VisitorTracker.update()` assigns stable IDs across frames. Test: `test_two_overlapping_faces_get_distinct_ids` passes (`[1, 2]`). | **Implemented** |
| **Visitor identity association** | Track has `visitor_id`, faces have `face.visitor_id` assigned. Integration with `FaceRecognizer` carries name/score. | **Implemented** |
| **Arrival event** | No event published by tracker itself. Events (`visitor.detected`) generated by `VisionService._publish_events()` which carries `visitor_id`, `name`, `status`, `score`. | **Partial** — events exist but not autonomously generated by tracker |
| **Presence state** | `track.missing_frames` counter; `_cleanup()` removes tracks after `max_missing_frames`. `active_track_count()` reports current count. | **Partial** — no "presence" event or state machine |
| **Departure event** | No departure event. Expired tracks silently cleaned up. | **Not implemented** |
| **Grace period for temporary disappearance** | `_max_missing_frames` config (default 30). Tracks with `missing_frames > _max_missing_frames` expired. | **Implemented** at tracker level |
| **Return recognition within sessions** | Expired IDs not reused (`test_expired_ids_are_not_reused`). New track gets next monotonic ID. | **Implemented** at tracker level |
| **Duplicate-event suppression** | Within a frame, `_find_best_track` excludes already-matched tracks. `matched_tracks` set prevents reuse. | **Implemented** at frame level |
| **Multiple independent visitor sessions** | `clear()` resets `_next_visitor_id = 1` (new session). Within session IDs monotonic and never reused. | **Implemented** at tracker level |
| **Integration with existing event system** | `VisionService._publish_events()` publishes `visitor.detected` with full payload. `EventBus` is thread-safe with failing-subscriber protection. | **Implemented** |

### Summary

- **VisitorTracker** (component responsibility: "Which detected face belongs to which ongoing track?") — **IMPLEMENTED**. Provides stable ID assignment across frames, expiration/cleanup, duplicate suppression, and session-reset capability.

- **VisitorSessionManager** (component responsibility: "Is the visitor arriving, present, or departing?") — **NOT IMPLEMENTED**. No state machine, no arrival/departure events, no grace period event.

- **GreetingController** (component responsibility: "Should Greeto interact now?") — **NOT IMPLEMENTED**. Beyond scope of this audit.

- **Integration with event system** — **PARTIAL**. `VisionService` publishes `visitor.detected` events, but these are not autonomously generated by the tracker. The connection between tracker state and event emission lives in `VisionService`.

### Overall Assessment

**Visitor Tracking and Session Intelligence is PARTIALLY IMPLEMENTED.**

The lower-level tracking (face-to-track association, ID stability, expiration) is complete. The higher-level session intelligence (arrival/departure events, grace period handling, session state machine, multiple independent session support at event level) is **not implemented**. The originally planned Milestone 4 scope includes items that go beyond the current tracker implementation.

---

## 8. Code Changes Made

### Modified Files (28 files)

1. **`app/main.py`** — Added `TTSService` creation, injected as `tts_service=tts` into `ConversationManager`; passed `speech` widget to `MainWindow`

2. **`app/brain/conversation_manager.py`** — Removed internal `TTSService(container)` construction; added `tts_service` constructor parameter; uses injected `self.tts = tts_service`; added `event_bus.publish("conversation.processing", text=text)` and `event_bus.publish("conversation.error", message=str(error))`

3. **`app/core/config_manager.py`** — Added `PROJECT_ROOT` computation; made `get_value()` return `default` instead of raising `KeyError`; added `sorted()` to YAML glob

4. **`app/core/container.py`** — Added `PROJECT_ROOT` import; improved log directory resolution with `PROJECT_ROOT`; added commented logging config improvements

5. **`app/core/event_bus.py`** — Added `from __future__ import annotations`; added module-level `logger`; added try/except around callback execution with `logger.exception`; documentation improvements

6. **`app/core/logging_setup.py`** — Added `parents=True` to `mkdir()`; added `logger.propagate = False`; formatter unchanged

7. **`app/gui/camera_widget.py`** — PyQt5 import fix (was PySide6)

8. **`app/gui/main_window.py`** — Major refactor: PyQt5 imports, `KioskPresenter` integration, `CaptionWidget`, `KioskBackground`, `VisionPump`, diagnostics, escape/F11/F10 controls, layout changes

9. **`app/gui/robot_widget.py`** — PyQt5 import fix; state management changes

10. **`app/gui/theme.py`** — `KIOSK_STYLE` introduced; `STYLE` kept for backward compatibility

11. **`app/gui/caption_widget.py`** — New file (caption display widget)

12. **`app/gui/kiosk_background.py`** — New file (kiosk background widget)

13. **`app/gui/kiosk_presenter.py`** — New file (presenter class connecting event bus to UI)

14. **`app/gui/face_animation.py`** — New file (face animation state)

15. **`app/gui/face_renderer.py`** — New file (face rendering)

16. **`app/vision/camera_manager.py** — Camera format verification, auto-detection, warmup frames, reconnect logic

17. **`app/vision/embedding_model.py`** — Relevant changes (not detailed in diff)

18. **`app/vision/enrollment.py`** — Relevant changes (not detailed in diff)

19. **`app/vision/face_cache.py`** — Relevant changes (not detailed in diff)

20. **`app/vision/face_database.py`** — SQLite replacement for JSON; lazy initialization; legacy migration with visit-counter discard; corrupt database recovery; concurrent-safe with `threading.RLock`

21. **`app/vision/face_recognizer.py`** — Relevant changes (not detailed in diff)

22. **`app/vision/frame_grabber.py`** — Relevant changes (not detailed in diff)

23. **`app/vision/models.py`** — Relevant changes (not detailed in diff)

24. **`app/vision/similarity.py`** — Relevant changes (not detailed in diff)

25. **`app/vision/vision_service.py`** — Relevant changes (not detailed in diff)

26. **`app/vision/visitor_database.py`** — Complete rewrite: SQLite-backed, thread-safe, lazy initialization, legacy JSON migration, corrupt database recovery, schema versioning, `register()` returns record, `get()`, `list_visitors()`, `stats()`

27. **`app/vision/visitor_tracker.py`** — Dataclass slots removed (-> plain dataclass); `_find_best_track` added `excluded` parameter; `_cleanup` ID session reset; `active_track_count()`, `has_active_tracks()`, `get_track_ids()` added

28. **`config/camera.yaml`** — Enhanced with `backend`, `auto_detect`, `probe_max_index`, `warmup_frames`, `read_failures_before_reconnect`, `reconnect_interval_s`, `open_retry_interval_s`, `release_timeout_s`

29. **`config/vision.yaml`** — New file with recognition threshold (`0.55`), `min_detection_score`, `min_face_size`, `expected_embedding_size`, `min_embedding_norm`, `normalize_embeddings`; visitor statistics config pointing to SQLite

### New Files (18 untracked)

- `app/core/admin_auth.py` — PIN authentication
- `app/gui/caption_widget.py` — Caption display
- `app/gui/face_animation.py` — Face animation state
- `app/gui/face_renderer.py` — Face rendering
- `app/gui/kiosk_background.py` — Kiosk background
- `app/gui/kiosk_presenter.py` — Kiosk presenter
- `app/gui/resources/robot.svg` — Robot SVG resource
- `app/gui/vision_pump.py` — Vision processing pump
- `app/vision/recognition.py` — Recognition configuration
- `config/admin.yaml` — Admin configuration
- `docs/` — Documentation directory
- `requirements_jetson.txt` — Jetson dependencies
- `requirements_windows.txt` — Windows dependencies
- `scripts/` — Script directory
- `tests/` — Test directory (with all current tests)

### Deleted Files (5)

- `app/test_database.py` — Simple test script, behavior covered by `tests/test_storage.py`
- `app/test_face_recognizer.py` — Face recognition test, behavior covered by `tests/test_recognition.py`
- `app/test_speech.py` — Speech service starter, behavior covered by `tests/test_speech.py`
- `app/test_vision.py` — Vision service starter, behavior covered by `tests/test_recognition.py`/`tests/test_face.py`
- `data/visitors.json` — Legacy JSON; migrated to SQLite

---

## 9. Test Commands and Actual Results

### test_speech.py

```
Command: python tests/test_speech.py
Exit code: 0
Checks: 4 passed, 0 failed
- A1. shared TTS serializes concurrent calls: PASS
- A2. empty text is ignored: PASS
- A3. Piper failure is reported and cleaned up: PASS
- B1. ConversationManager uses the injected TTS: PASS
```

### test_storage.py

```
Command: python tests/test_storage.py
Exit code: 0
Checks: 23 passed, 0 failed
- A1. empty database is created lazily: PASS
- A2. repeated visits are counted: PASS
- A3. records persist across instances: PASS
- A4. names are validated and tidied: PASS
- B1. legacy identities import, old stats discarded: PASS
- B2. invalid legacy file is archived: PASS
- C1. corrupt database is archived: PASS
- C2. newer schema is refused: PASS
- C3. temporary database is isolated: PASS
- D1. concurrent registration loses no visits: PASS (40 writes from 40 threads)
```

### test_enrollment.py

```
Command: python tests/test_enrollment.py
Exit code: 0
Checks: 34 passed, 0 failed
- A1. identity names are validated: PASS
- B1. valid enrolment writes images and cache: PASS
- B2. duplicate source paths are collapsed: PASS
- B3. adding and replacing photos: PASS
- C1. bad batches write nothing: PASS
- C2. bad replacement keeps existing photos: PASS
- C3. missing and undecodable sources: PASS
- D1. listing and removal: PASS
- E1. PIN hashing and verification: PASS
- E2. PIN config round trip: PASS
- F1. CLI set-pin and auth gate: PASS
- F2. CLI list and visitors: PASS
```

### test_recognition.py

```
Command: python tests/test_recognition.py
Exit code: 0 (exit code shown as "0" at end, all 35 checks passed)
Checks: 35 passed, 0 failed
- A1. embedding validation and normalisation: PASS
- A2. cosine similarity guards: PASS
- B1. config defaults, overrides and fallbacks: PASS
- C1. recognized vs unknown results: PASS
- C2. detection/size quality gates: PASS
- C3. invalid embeddings: PASS
- C4. empty and broken databases: PASS
- C5. batch never aborts: PASS
- C6. configurable threshold decision: PASS
- C7. stats expose threshold, database size and last outcome: PASS
- D1. overlapping faces get unique IDs: PASS
- D2. many faces at one centroid: PASS
- D3. expired IDs are never reused: PASS
- D4. IDs survive movement: PASS
- E1. calibration with clear separation: PASS
- E2. calibration with overlap: PASS
- E3. calibration refuses to guess without impostors: PASS
- F1. malformed cached embeddings are dropped: PASS
- F2. database cache fallbacks: PASS
- G1. structured visitor events: PASS
- G2. health() reports camera, grabber, recognition and visitors: PASS
```

---

## 10. Hardware Tests Not Performed

- No camera hardware available in audit environment
- No InsightFace model (`models/buffalo_l`) loaded — all recognition tests use numpy fakes
- No Piper TTS model loaded — speech tests use placeholder `voice.onnx`
- No microphone for Whisper STT
- No actual Qt display (tests run headless)
- No Jetson-specific optimizations verified

---

## 11. Remaining Known Limitations

1. **Visitor Session State Machine**: No formal "arrival/present/departure" state machine. Tracker provides ID stability but not session lifecycle events.

2. **Grace Period Event Emission**: While `_max_missing_frames` config exists, there is no `departure` event published when a track expires.

3. **Multiple Independent Session Event Separation**: The event bus publishes `visitor.detected` but does not distinguish which session a visitor belongs to when multiple visitors are present.

4. **Return Recognition Workflow**: No special handling for returning visitors beyond ID stability. No "welcome back" logic.

5. **Personalized Greeting**: The `Hello! Saksham` greeting remains for a later phase (greeting-intelligence).

6. **No Camera in Audit Environment**: All visitor tracking tests use synthetic faces; camera integration not verified in this audit.

---

## 12. Recommended Next Steps

### For TTS Integration (Complete)

- TTS single-instance ownership is unambiguous and verified.
- Concurrent speech safety is verified under tested conditions.
- Piper failure and cleanup paths are verified.
- Speech event behavior is correct.
- No further TTS work needed for Phase 3 Milestone 4 scope.

### For Visitor Tracking and Session Intelligence (Partial — Remaining Work)

1. **Implement VisitorSessionManager** — Add state machine with `arriving`, `present`, `departing` states.
2. **Add Departure Event** — Publish `visitor.departed` when track expires after grace period.
3. **Add Arrival Event** — Publish `visitor.arrived` when a new track is first created (not just `visitor.detected`).
4. **Session Separation** — Ensure event payloads include session identifiers for multiple independent visitors.
5. **Return Recognition Workflow** — Implement logic to recognize returning visitors and emit appropriate events.
6. **Camera Integration** — Connect face detection pipeline to visitor tracker in real time.

### Documentation

- Update `docs/PHASE_3_MILESTONE_4_AUDIT_REPORT.md` with any new findings after next phase work.
- Ensure `.gitignore` does not hide runtime files needed for development.

### Acceptance Criteria Status

- ✅ Actual repository state inspected
- ✅ Every material claim in previous report independently checked
- ✅ Incorrect implementation claims corrected (Milestone 4 ≠ TTS + cleanup)
- ✅ All verified in-scope bugs fixed (none found — all tests pass)
- ✅ TTS ownership unambiguous
- ✅ Concurrent speech safe under tested conditions
- ✅ Piper failure and cleanup paths verified
- ✅ Speech event behavior correct
- ✅ Deleted files checked for dependencies
- ✅ Phase 3 Milestone 3 functionality remains intact
- ✅ Tests independently executed and accurately reported
- ✅ Documentation reflects actual implementation
- ✅ No real visitor data changed
- ✅ Originally planned Visitor Tracking milestone assessed separately (partial)

---