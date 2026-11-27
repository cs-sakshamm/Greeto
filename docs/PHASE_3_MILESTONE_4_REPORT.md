# Phase 3 — Milestone 4 Implementation Report

**Scope:** shared speech output and repository cleanup.
**Status:** complete; all targeted and regression checks pass.
**Date:** 23 November 2026.

## 1. What changed

### Files modified

| File | Change |
| --- | --- |
| `app/speech/tts_service.py` | Added a lock around the complete generate-and-play cycle; optional `model_path` for tests |
| `app/brain/conversation_manager.py` | No longer constructs TTS; accepts the application's shared `tts_service` |
| `app/main.py` | Creates one `TTSService` at startup and injects it into `ConversationManager` |

### Files created

| File | Purpose |
| --- | --- |
| `tests/test_speech.py` | 4 hardware-free checks for serialized speech, failure cleanup and injection |
| `docs/PHASE_3_MILESTONE_4_REPORT.md` | This report |

### Files deleted

| File | Reason |
| --- | --- |
| `app/test_database.py` | One-off manual script, unreferenced, superseded by `tests/test_storage.py` and `tests/test_recognition.py` |
| `app/test_face_recognizer.py` | One-off manual script, unreferenced, superseded by `tests/test_face.py` and `tests/test_recognition.py` |
| `app/test_speech.py` | One-off manual script that ran forever, unreferenced, superseded by `tests/test_speech.py` |
| `app/test_vision.py` | One-off manual script, unreferenced, superseded by `tests/test_camera.py` |

No dependency was added. No model, face, configuration, visitor
database or knowledge file was changed.

## 2. Decisions

1. **Startup owns the single TTS instance.** `app/main.py` constructs
   `TTSService` once and passes it to `ConversationManager`. The manager
   no longer creates a private instance, so there is one owner and one
   lifetime for speech output.
2. **Injection is explicit and required.** `ConversationManager` takes
   `tts_service` as a constructor argument. A caller cannot accidentally
   create a second TTS service by forgetting to pass one.
3. **Serialization belongs in the shared service.** A normal `Lock`
   covers Piper generation plus playback in `speak()`. If two
   components eventually speak at once, one waits instead of both
   playing over each other. The utterance counter moves under the same
   lock, so event numbering matches playback order.
4. **Lifecycle events are unchanged.** `speech.speaking_started`,
   `speech.speaking_finished` and `speech.error` keep the same names,
   payload fields and ordering. The kiosk needs no changes.
5. **A test seam, not a new deployment option.** `TTSService` accepts an
   optional `model_path`; the application still uses the fixed project
   model. This lets tests run without loading the 63 MB voice model.
6. **No real audio in tests.** The tests replace `subprocess.Popen` with
   a fake Piper process and `_play_audio` with a thread-safe probe. They
   exercise the real lock, events, cleanup and error path without a
   speaker, microphone, model load or emitted sound.
7. **Delete only what is provably unused.** The four `app/test_*.py`
   scripts had no imports or documentation references and duplicated
   behaviour already covered by `tests/`. Other empty placeholder
   modules and unused assets were left untouched because removing them
   was not part of this milestone.

## 3. Tests executed and actual results

Environment: Windows, `.venv` Python 3.11.1; suites run serially after
the final edits.

| Command | Result |
| --- | --- |
| `python -m compileall -q app scripts tests` | **Exit 0** |
| `python tests/test_speech.py` | **ALL CHECKS PASSED** (4/4) |
| `python tests/test_storage.py` | **ALL CHECKS PASSED** (10/10) |
| `python tests/test_enrollment.py` | **ALL CHECKS PASSED** (12/12) |
| `python tests/test_recognition.py` | **ALL CHECKS PASSED** (21/21) |
| `python tests/test_camera.py` | **ALL CHECKS PASSED** (9 passed; live skipped) |
| `python tests/test_face.py --skip-slow` | **ALL CHECKS PASSED** (10 passed; long run skipped) |
| `python tests/test_kiosk.py` | **ALL CHECKS PASSED** (13 passed; continuous run skipped) |

### Speech checks

| Check | What it proves |
| --- | --- |
| A1. shared TTS serializes concurrent calls | Two threads calling the same service produce two utterances, never overlapping playback; both WAVs are deleted; started/finished events match |
| A2. empty text is ignored | No utterance number, process, event or file is produced |
| A3. Piper failure is reported and cleaned up | Exit code 3 produces `speech.error(kind="tts")`, no playback and no leaked WAV |
| B1. ConversationManager uses the injected TTS | A stub TTS receives the parsed answer once; the real manager is used with stubbed LLM/knowledge/prompt only |

A1 used a `threading.Barrier` so both calls definitely attempted speech
together; the observed maximum concurrent playback was **1**.

## 4. Existing functionality preserved

- Event names and payloads consumed by `KioskPresenter` are unchanged.
- `speech.speaking_started` is still published only after Piper has
  produced the WAV and immediately before playback.
- `speech.speaking_finished` is still published in `finally`, including
  failed playback.
- The kiosk, face animation, camera, recognition, storage and enrolment
  suites all pass unchanged.
- No change to recognition thresholds, tracking, greeting, visitor
  records, Whisper, the local LLM or the knowledge base.
- No new dependency and no change to the model locations.

## 5. Known limitations and unresolved risks

1. **Real audible playback was not exercised here.** The test isolates
   the service with a fake Piper process and fake player. The real
   speaker path remains a hardware check.
2. **TTS is still required at startup.** A missing voice model or
   `piper` executable still aborts startup through the existing fatal
   path. This milestone shares and serializes TTS; it does not add
   voice-only degraded mode.
3. **No interruption or stop control.** A long utterance cannot be
   cancelled; shutdown does not explicitly stop an in-flight Piper
   process. The conversation thread is daemonized as before.
4. **Sharing has one consumer today.** `ConversationManager` is the only
   component that speaks. The ownership and locking are ready for a
   future greeting or status voice, but no new caller was added.
5. **The lock is process-local.** It does not coordinate multiple
   application processes. The kiosk remains a single-process design.
6. **No real model load in the new tests.** `TTSService` construction
   with the fixed model path is covered by the application path, but the
   new suite uses a temporary placeholder.

## 6. Repository state note

Before this milestone, the working tree already contained a real
migration of `data/visitors.json`: `data/visitors.db` and
`data/visitors.json.migrated` were present, and the legacy file was
removed. Those are runtime files ignored by Git. Milestone 4 did not
touch or re-run that migration.

## 7. Recommendation

Milestone 4 is complete: the application now creates exactly one TTS
service at startup, passes it explicitly to the conversation layer,
serializes concurrent speech, and the stale manual scripts are removed.
All targeted and regression checks actually run here pass.

Phase 3 has no remaining approved work item. I recommend defining the
next phase before changing more code. The natural next candidates are
voice-only degraded mode for a missing TTS model, real speaker testing
on the target hardware, or the next product milestone for the reception
flow.

**No next milestone has been started.** I will wait for the next scope.
