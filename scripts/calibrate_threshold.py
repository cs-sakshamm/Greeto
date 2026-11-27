"""
Estimate a recognition threshold from the enrolled faces.

Genuine pairs are two photos of the same person; impostor pairs are
photos of different people. The script prints the score distributions
and the threshold with the best balanced accuracy. It never edits the
configuration: review the number and update ``config/vision.yaml``
yourself.

Usage:

    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --max-pairs 50000
    python scripts/calibrate_threshold.py --low 0.2 --high 0.9

The result is only meaningful with several photos per person and at
least two enrolled people. With one photo per person there are no
genuine pairs, and with one person there are no impostor pairs.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

from app.core.config_manager import ConfigManager  # noqa: E402
from app.vision.embedding_model import EmbeddingModel  # noqa: E402
from app.vision.face_database import FaceDatabase  # noqa: E402
from app.vision.recognition import (  # noqa: E402
    RecognitionConfig,
    calibrate_threshold,
)
from app.vision.similarity import (  # noqa: E402
    cosine_similarity,
    normalize_embedding,
)


def _describe(scores):

    if not scores:

        return "n/a"

    ordered = sorted(scores)

    mean = sum(ordered) / len(ordered)

    return "min %.4f  mean %.4f  max %.4f" % (
        ordered[0],
        mean,
        ordered[-1],
    )


def _collect_pairs(database, max_pairs):

    genuine = []
    impostor = []

    normalised = {}

    rejected = 0

    for name, embeddings in database.items():

        vectors = []

        for embedding in embeddings:

            try:

                vectors.append(normalize_embedding(embedding))

            except ValueError:

                rejected += 1

        if vectors:

            normalised[name] = vectors

    for vectors in normalised.values():

        for first, second in itertools.combinations(vectors, 2):

            genuine.append(cosine_similarity(first, second))

    names = sorted(normalised)

    exhausted = False

    for first_name, second_name in itertools.combinations(names, 2):

        for first in normalised[first_name]:

            for second in normalised[second_name]:

                impostor.append(cosine_similarity(first, second))

                if len(impostor) >= max_pairs:

                    exhausted = True

                    break

            if exhausted:

                break

        if exhausted:

            break

    return genuine, impostor, len(normalised), rejected, exhausted


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Calibrate the face recognition threshold."
    )

    parser.add_argument(
        "--low",
        type=float,
        default=0.0,
        help="lowest threshold to test (default 0.0)",
    )

    parser.add_argument(
        "--high",
        type=float,
        default=1.0,
        help="highest threshold to test (default 1.0)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1001,
        help="threshold grid size (default 1001)",
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=50000,
        help="cap on cross-identity pairs (default 50000)",
    )

    args = parser.parse_args(argv)

    config = ConfigManager()

    try:

        config.load()

    except Exception as error:  # noqa: BLE001 - diagnostic script

        print("Could not load config: %s" % error)

        return 1

    current = RecognitionConfig.from_config(config)

    print("Greeto threshold calibration")
    print("=" * 64)
    print("current threshold : %.3f" % current.threshold)

    for warning in current.warnings:

        print("config warning    : %s" % warning)

    print()
    print("Loading face database (this loads the InsightFace model)...")

    database = FaceDatabase(EmbeddingModel())

    database.load()

    mapping = database.get_database()

    genuine, impostor, identities, rejected, capped = _collect_pairs(
        mapping,
        max(args.max_pairs, 1),
    )

    print("identities        : %d" % identities)
    print("embeddings        : %d" % database.embedding_count())
    print("invalid skipped   : %d" % rejected)
    print("genuine pairs     : %d" % len(genuine))
    print("impostor pairs    : %d%s"
          % (len(impostor), " (capped)" if capped else ""))
    print("genuine scores    : %s" % _describe(genuine))
    print("impostor scores   : %s" % _describe(impostor))
    print()

    if not genuine or not impostor:

        print("Not enough data to calibrate safely.")
        if not genuine:
            print("- need at least two photos of the same person "
                  "(currently one photo per identity)")
        if not impostor:
            print("- need at least two enrolled people")
        print()
        print("Keeping the configured threshold of %.3f." % current.threshold)
        print("Add more photos/people, then run this script again.")
        return 0

    result = calibrate_threshold(
        genuine,
        impostor,
        low=args.low,
        high=args.high,
        steps=args.steps,
    )

    if result is None:

        print("Calibration returned no result; keeping %.3f."
              % current.threshold)
        return 0

    print("recommended threshold : %.3f" % result.threshold)
    print("balanced accuracy     : %.3f" % result.balanced_accuracy)
    print("false accept rate     : %.3f" % result.false_accept_rate)
    print("false reject rate     : %.3f" % result.false_reject_rate)
    print()
    print("Genuine/impostor counts: %d / %d"
          % (result.genuine_count, result.impostor_count))
    print("This is an estimate from enrolled photos, not live-camera data.")
    print("To apply it, set vision.recognition.threshold in "
          "config/vision.yaml to %.2f and restart Greeto."
          % result.threshold)

    return 0


if __name__ == "__main__":

    sys.exit(main())
