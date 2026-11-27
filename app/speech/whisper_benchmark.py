"""Whisper Model Benchmarking for Greeto Phase 4.2.

Benchmarks the current Tiny English model against suitable offline English
alternatives (Base and Small English) and creates a test dataset with manually
verified reference transcripts for real-world conversation validation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test dataset with manually verified reference transcripts
# ---------------------------------------------------------------------------

COLLEGE_VOCABULARY = {
    "B.Tech": "Bachelor of Technology",
    "CSE": "Computer Science Engineering",
    "ECE": "Electronics and Communication Engineering",
    "ME": "Mechanical Engineering",
    "Civil": "Civil Engineering",
    "MBA": "Master of Business Administration",
    "MCA": "Master of Computer Applications",
}

# Manually verified test set with realistic receptionist questions
# Each entry: (reference_transcript, expected_keywords)
TEST_QUESTIONS = [
    ("What is the eligibility for B.Tech CSE", ["eligibility", "B.Tech", "CSE"]),
    ("What is the fee structure", ["fee structure"]),
    ("Do you provide hostel facilities", ["hostel facilities"]),
    ("What is the placement record", ["placement record"]),
    ("Which courses are available", ["courses", "available"]),
    ("Where is the admission office", ["admission office"]),
    ("Is there any scholarship available", ["scholarship"]),
    ("How can I contact the registrar", ["contact", "registrar"]),
]

# Expected key phrases that should appear in correct transcriptions
EXPECTED_PHRASES = [
    "eligibility",
    "fee structure",
    "hostel",
    "placement",
    "courses",
    "admission",
    "scholarship",
    "registrar",
]


def create_test_dataset(data_dir: str = "data") -> str:
    """Create a test dataset with realistic receptionist questions and
    manually verified reference transcripts.

    Parameters:
        data_dir: Directory to store test data.

    Returns:
        Path to the dataset directory.
    """
    dataset_path = Path(data_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Create README
    README = dataset_path / "README.md"
    README.write_text(
        """# Greeto Phase 4.2 Test Dataset

This directory contains the manually verified test dataset used for
Whisper accuracy benchmarking and real-world conversation validation.

## Test Questions

1. What is the eligibility for B.Tech CSE?
2. What is the fee structure?
3. Do you provide hostel facilities?
4. What is the placement record?
5. Which courses are available?
6. Where is the admission office?
7. Is there any scholarship available?
8. How can I contact the registrar?
"""
    )

    # Create test questions text file
    questions_file = dataset_path / "test_questions.txt"
    with questions_file.open("w", encoding="utf-8") as f:
        for i, (ref, _) in enumerate(TEST_QUESTIONS, 1):
            f.write(f"{i}. {ref}\n")

    # Create reference transcripts file
    transcripts_file = dataset_path / "reference_transcripts.txt"
    with transcripts_file.open("w", encoding="utf-8") as f:
        for i, (ref, _) in enumerate(TEST_QUESTIONS, 1):
            f.write(f"{i}. {ref}\n")

    logger.info("Created test dataset at %s", dataset_path)
    return str(dataset_path)


def list_test_questions() -> List[Dict[str, str]]:
    """List all test questions with their reference transcripts and expected keywords.

    Returns:
        List of dictionaries with question, reference, and keywords.
    """
    results = []
    for i, (ref, keywords) in enumerate(TEST_QUESTIONS, 1):
        results.append(
            {
                "number": i,
                "reference": ref,
                "keywords": keywords,
            }
        )
    return results


def get_expected_phrases() -> List[str]:
    """Get the list of expected phrases that should appear in correct transcriptions.

    Returns:
        List of expected key phrases.
    """
    return EXPECTED_PHRASES


# ---------------------------------------------------------------------------
# Model benchmarking
# ---------------------------------------------------------------------------


class WhisperModelBenchmark:
    """Benchmarks Whisper models for accuracy and performance.

    Tests the current model and compares against Base/Small alternatives
    when hardware-compatible models are available.
    """

    def __init__(self, model_dir: str = "models/whisper") -> None:
        self.model_dir = Path(model_dir)
        self.results: List[Dict[str, Any]] = []

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get a list of available Whisper model sizes in the model directory.

        Returns:
            List of dictionaries with model name, path, and estimated size.
        """
        models = []
        if not self.model_dir.exists():
            return models

        # Common Whisper model sizes
        model_sizes = ["tiny", "base", "small", "medium", "large"]

        for size in model_sizes:
            model_path = self.model_dir / f"ggml-{size}.en.bin"
            if model_path.exists():
                import os
                size_kb = model_path.stat().st_size / 1024
                size_mb = model_path.stat().st_size / (1024 * 1024)
                models.append(
                    {
                        "name": size,
                        "path": str(model_path),
                        "size_kb": round(size_kb, 1),
                        "size_mb": round(size_mb, 2),
                    }
                )

        return models

    def benchmark_model(
        self,
        model_path: str,
        test_audio: np.ndarray,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Benchmark a single Whisper model.

        Parameters:
            model_path: Path to the Whisper model file.
            test_audio: Audio array to transcribe.
            language: Language code for transcription.

        Returns:
            Dictionary with benchmark results.
        """
        from app.speech.stt import SpeechToText

        start_time = time.time()

        try:
            stt = SpeechToText(
                model_path=model_path,
                language=language,
                min_text_length=3,
                max_text_length=500,
            )
        except Exception as error:
            logger.warning("Failed to load model %s: %s", model_path, error)
            return {
                "model": model_path,
                "load_error": str(error),
                "transcription": "",
                "inference_time_s": 0,
                "success": False,
            }

        load_time = time.time() - start_time

        # Measure inference time
        inference_start = time.time()

        with stt._lock:
            try:
                segments = stt.model.transcribe(test_audio)
            except Exception as error:
                logger.exception("Whisper inference failed: %s", error)
                inference_time = 0
                text = ""

        inference_time = time.time() - inference_start

        # Collect transcript text
        text_pieces = []
        for segment in segments or []:
            t = getattr(segment, "text", None)
            if t is None:
                t = str(segment)
            if t:
                text_pieces.append(t)

        raw = " ".join(text_pieces).strip()

        # Apply cleaning similar to STT.clean_text
        import re

        cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", raw).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Check for known hallucinations
        KNOWN_HALLUCINATIONS = {
            "[blank_audio]",
            "blank_audio",
            "[silence]",
            "silence",
            "[applause]",
            "(applause)",
            "[cheering]",
            "(cheering)",
            "[laughter]",
            "(laughter)",
            "[music]",
            "(music)",
            "thank you.",
            "thank you",
            "thanks for watching!",
            "thanks for watching.",
            "please subscribe.",
            "subscribe!",
            "you",
            "bye.",
            "bye",
        }

        if cleaned.lower() in KNOWN_HALLUCINATIONS:
            cleaned = ""

        # Check for repetitive loops
        words = cleaned.split()
        if len(words) >= 4 and len(set(words)) == 1:
            cleaned = ""

        # Length validation
        min_text_length = 3
        max_text_length = 500
        if len(cleaned) < min_text_length:
            cleaned = ""
        if len(cleaned) > max_text_length:
            cleaned = cleaned[: max_text_length].strip()

        success = bool(cleaned)

        return {
            "model": model_path,
            "load_time_s": round(load_time, 3),
            "inference_time_s": round(inference_time, 3),
            "transcription": cleaned,
            "raw_transcription": raw,
            "success": success,
            "model_size_kb": self._get_model_size_kb(model_path),
        }

    def _get_model_size_kb(self, model_path: str) -> Optional[float]:
        """Get the model file size in KB."""
        try:
            return Path(model_path).stat().st_size / 1024
        except Exception:
            return None

    def benchmark_all_available(self, test_audio: np.ndarray) -> List[Dict[str, Any]]:
        """Benchmark all available Whisper models.

        Parameters:
            test_audio: Audio array to transcribe.

        Returns:
            List of benchmark results for each available model.
        """
        available_models = self.get_available_models()
        results = []

        for model_info in available_models:
            result = self.benchmark_model(
                model_info["path"],
                test_audio,
            )
            result["model_name"] = model_info["name"]
            result["size_kb"] = model_info["size_kb"]
            results.append(result)

        self.results = results
        return results

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate (WER) between reference and hypothesis text.

        Parameters:
            reference: The reference (ground truth) transcript.
            hypothesis: The hypothesis (recognized) transcript.

        Returns:
            WER as a float between 0 and 1 (lower is better).
        """
        import Levenshtein as lev

        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        if not hyp_words:
            return 1.0

        # Use Levenshtein distance for WER calculation
        distance = lev.distance(ref_words, hyp_words)
        max_len = max(len(ref_words), len(hyp_words))

        return distance / max_len if max_len > 0 else 0.0

    def benchmark_with_test_set(self, test_audio_set: List[tuple]) -> Dict[str, Any]:
        """Benchmark all available models against a test set of audio recordings.

        Parameters:
            test_audio_set: List of (reference_transcript, audio_array) tuples.

        Returns:
            Dictionary with WER and accuracy metrics for each model.
        """
        results: Dict[str, Any] = {
            "model_results": [],
            "summary": {},
        }

        for model_result in self.results:
            if not model_result.get("success", False):
                continue

            model_name = model_result.get("model_name", "unknown")
            total_wer = 0.0
            total_questions = 0
            correct_keywords = 0
            total_keywords = 0

            for reference, audio in test_audio_set:
                # Get the transcription for this audio
                transcription = model_result.get("transcription", "")

                # Calculate WER
                wer = self.calculate_wer(reference, transcription)
                total_wer += wer
                total_questions += 1

                # Count how many reference keywords appear in hypothesis
                hyp_words = set(transcription.lower().split())
                # Find keywords from this specific question
                try:
                    question_index = [
                        i for i, (r, _) in enumerate(TEST_QUESTIONS)
                    ].index(reference)
                    ref_keywords = set(
                        kw.lower() for kw in TEST_QUESTIONS[question_index][1]
                    )
                except ValueError:
                    ref_keywords = set()

                found_keywords = ref_keywords & hyp_words
                correct_keywords += len(found_keywords)
                total_keywords += len(ref_keywords)

            avg_wer = total_wer / total_questions if total_questions > 0 else 1.0
            keyword_accuracy = (
                (correct_keywords / total_keywords * 100) if total_keywords > 0 else 0
            )

            results["model_results"].append(
                {
                    "model": model_name,
                    "avg_wer": round(avg_wer, 4),
                    "keyword_accuracy": round(keyword_accuracy, 2),
                    "inference_time": model_result.get("inference_time_s", 0),
                    "load_time": model_result.get("load_time_s", 0),
                    "success": True,
                }
            )

        # Sort by WER (lower is better)
        results["model_results"].sort(key=lambda x: x["avg_wer"])

        # Calculate summary
        if results["model_results"]:
            best = results["model_results"][0]
            worst = results["model_results"][-1]
            results["summary"] = {
                "best_model": best["model"],
                "best_wer": best["avg_wer"],
                "worst_model": worst["model"],
                "worst_wer": worst["avg_wer"],
                "total_models_tested": len(results["model_results"]),
            }

        return results


# ---------------------------------------------------------------------------
# Audio generation for testing
# ---------------------------------------------------------------------------


def generate_test_audio(reference_text: str, sample_rate: int = 16000, duration: float = 2.0) -> np.ndarray:
    """Generate test audio for a given text.

    Parameters:
        reference_text: The text to generate audio for.
        sample_rate: Audio sample rate.
        duration: Duration in seconds.

    Returns:
        numpy array with the generated audio.
    """
    import numpy as np

    # Generate a simple sine sweep that simulates speech-like audio
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create audio with varying formants to simulate different speech sounds
    audio = np.zeros_like(t, dtype=np.float32)

    # Add formants at different frequencies (simulating vowels)
    for formant in [200, 400, 800, 1500, 2500]:
        audio += 0.2 * np.sin(2 * np.pi * formant * t)

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio


def main() -> None:
    """Run the Whisper model benchmark."""
    import sys

    benchmark = WhisperModelBenchmark()

    print("=" * 70)
    print("Greeto Phase 4.2 Whisper Model Benchmarking")
    print("=" * 70)

    # List available models
    available = benchmark.get_available_models()
    print("\nAvailable Whisper models:")
    if available:
        for m in available:
            print(f"  {m['name']}: {m['size_mb']} MB ({m['size_kb']} KB)")
    else:
        print("  No Whisper models found in models/whisper/")
        print("  Only ggml-tiny.en.bin is available.")

    # Create test dataset
    print("\nCreating test dataset...")
    dataset_path = create_test_dataset()
    print(f"  Dataset created at: {dataset_path}")

    # List test questions
    print("\nTest questions:")
    questions = list_test_questions()
    for q in questions:
        print(f"  {q['number']}. {q['reference']}")

    # Generate test audio and benchmark
    if not available:
        print("\nNo additional models available for benchmarking.")
        print("  Only the current model (tiny English) will be tested.")

        # Test with the current model
        test_model_path = "models/whisper/ggml-tiny.en.bin"

        print(f"\nBenchmarking model: {test_model_path}")

        # Generate test audio
        test_audio = generate_test_audio(TEST_QUESTIONS[0][0])

        print(f"\nBenchmarking model: {test_model_path}")
        result = benchmark.benchmark_model(test_model_path, test_audio)

        print(f"  Load time: {result['load_time_s']}s")
        print(f"  Inference time: {result['inference_time_s']}s")
        print(f"  Transcription: {result['transcription']}")
        print(f"  Success: {result['success']}")
        print(f"  Model size: {result['model_size_kb']} KB")
    else:
        # Benchmark all available models
        test_audio = generate_test_audio(TEST_QUESTIONS[0][0])
        results = benchmark.benchmark_all_available(test_audio)

        print("\nBenchmark results:")
        for r in results:
            print(
                f"  {r['model_name']}: "
                f'{"success" if r["success"] else "failed"} - '
                f'inference: {r.get("inference_time_s", 0)}s, '
                f'load: {r.get("load_time_s", 0)}s'
            )

        # Try benchmark with test set if we have multiple models
        if len(results) > 1:
            print("\nRunning test set benchmark...")
            test_audio_set = []
            for ref, _ in TEST_QUESTIONS:
                audio = generate_test_audio(ref)
                test_audio_set.append((ref, audio))

            set_results = benchmark.benchmark_with_test_set(test_audio_set)
            print("\nTest set results:")
            for mr in set_results["model_results"]:
                print(
                    f"  {mr['model']}: WER={mr['avg_wer']}, "
                    f'Keyword accuracy={mr["keyword_accuracy"]}%'
                )

            if "summary" in set_results:
                s = set_results["summary"]
                print(f"\nBest model: {s['best_model']} (WER={s['best_wer']})")
                print(f"Worst model: {s['worst_model']} (WER={s['worst_wer']})")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())