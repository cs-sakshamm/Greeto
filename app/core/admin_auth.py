"""
Admin authentication for the enrolment CLI.

The PIN is never stored in plain text. A PBKDF2-HMAC-SHA256 hash with a
random 16-byte salt is written to ``config/admin.yaml``:

    pin_hash: pbkdf2_sha256$210000$<salt>$<digest>

The application itself never reads this file; it exists only for the
offline enrolment tools. Verification uses a constant-time comparison.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from pathlib import Path

import yaml


PIN_ALGORITHM = "pbkdf2_sha256"

PIN_ITERATIONS = 210_000

MIN_ITERATIONS = 100_000

MAX_ITERATIONS = 2_000_000

MIN_PIN_LENGTH = 4

MAX_PIN_LENGTH = 12

PIN_HASH_KEY = "pin_hash"


class AdminAuthError(ValueError):
    """Raised when a PIN or admin configuration is invalid."""


def validate_pin(pin: str) -> str:
    """Validate a numeric PIN and return it unchanged."""

    if not isinstance(pin, str):

        raise AdminAuthError("PIN must be text")

    if not pin:

        raise AdminAuthError("PIN is empty")

    if not pin.isdigit():

        raise AdminAuthError("PIN must contain digits only")

    if len(pin) < MIN_PIN_LENGTH or len(pin) > MAX_PIN_LENGTH:

        raise AdminAuthError(
            "PIN must be %d-%d digits"
            % (MIN_PIN_LENGTH, MAX_PIN_LENGTH)
        )

    return pin


def _b64encode(value: bytes) -> str:

    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:

    padding = "=" * (-len(value) % 4)

    return base64.urlsafe_b64decode(value + padding)


def hash_pin(
    pin: str,
    *,
    salt: bytes | None = None,
    iterations: int = PIN_ITERATIONS,
) -> str:
    """Return a salted PBKDF2 hash string for ``pin``."""

    validated = validate_pin(pin)

    if not isinstance(iterations, int):

        raise AdminAuthError("PIN iteration count must be an integer")

    if iterations < MIN_ITERATIONS or iterations > MAX_ITERATIONS:

        raise AdminAuthError(
            "PIN iteration count must be %d-%d"
            % (MIN_ITERATIONS, MAX_ITERATIONS)
        )

    if salt is None:

        salt = secrets.token_bytes(16)

    if not isinstance(salt, bytes) or len(salt) < 8:

        raise AdminAuthError("PIN salt must be at least 8 bytes")

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        validated.encode("utf-8"),
        salt,
        iterations,
    )

    return "%s$%d$%s$%s" % (
        PIN_ALGORITHM,
        iterations,
        _b64encode(salt),
        _b64encode(digest),
    )


def verify_pin(pin: str, encoded: str) -> bool:
    """Check ``pin`` against an encoded hash.

    Malformed hashes and invalid PINs return False rather than raising,
    so a damaged configuration cannot be mistaken for a successful
    login.
    """

    if not isinstance(encoded, str) or not encoded:

        return False

    try:

        algorithm, iterations_text, salt_text, digest_text = encoded.split(
            "$"
        )

    except ValueError:

        return False

    if algorithm != PIN_ALGORITHM:

        return False

    try:

        iterations = int(iterations_text)

        salt = _b64decode(salt_text)

        expected = _b64decode(digest_text)

        pin = validate_pin(pin)

    except (ValueError, AdminAuthError):

        return False

    if iterations < MIN_ITERATIONS or iterations > MAX_ITERATIONS:

        return False

    if len(salt) < 8 or len(expected) < 16:

        return False

    actual = hashlib.pbkdf2_hmac(
        "sha256",
        pin.encode("utf-8"),
        salt,
        iterations,
    )

    return hmac.compare_digest(actual, expected)


def is_pin_hash(value: str) -> bool:
    """Return True when ``value`` looks like a supported PIN hash."""

    return isinstance(value, str) and value.startswith(
        PIN_ALGORITHM + "$"
    ) and len(value.split("$")) == 4


def load_pin_hash(path: str | Path) -> str:
    """Read the configured hash, returning an empty string if absent."""

    path = Path(path)

    if not path.exists():

        return ""

    try:

        data = yaml.safe_load(path.read_text(encoding="utf-8"))

    except (OSError, yaml.YAMLError) as error:

        raise AdminAuthError(
            "could not read admin configuration: %s" % error
        )

    if data is None:

        return ""

    if not isinstance(data, dict):

        raise AdminAuthError(
            "admin configuration must contain a mapping"
        )

    admin = data.get("admin", {})

    if admin is None:

        return ""

    if not isinstance(admin, dict):

        raise AdminAuthError("'admin' must contain a mapping")

    value = admin.get(PIN_HASH_KEY, "")

    if value is None:

        return ""

    if not isinstance(value, str):

        raise AdminAuthError("'pin_hash' must be text")

    return value.strip()


def save_pin_hash(path: str | Path, encoded: str) -> None:
    """Atomically write a PIN hash to the admin configuration."""

    if not is_pin_hash(encoded) or not verify_pin_format(encoded):

        raise AdminAuthError("refusing to write an invalid PIN hash")

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    text = (
        "# Admin PIN for scripts/enroll.py. Keep this file private: it\n"
        "# contains a salted hash, never the PIN itself.\n"
        "admin:\n"
        "  pin_hash: %s\n" % encoded
    )

    temporary.write_text(text, encoding="utf-8")

    os.replace(temporary, path)


def verify_pin_format(encoded: str) -> bool:
    """Validate the structure and iteration bounds of a hash string."""

    if not is_pin_hash(encoded):

        return False

    try:

        _, iterations_text, salt_text, digest_text = encoded.split("$")

        iterations = int(iterations_text)

        salt = _b64decode(salt_text)

        digest = _b64decode(digest_text)

    except (ValueError, TypeError):

        return False

    return (
        MIN_ITERATIONS <= iterations <= MAX_ITERATIONS
        and len(salt) >= 8
        and len(digest) >= 16
    )
