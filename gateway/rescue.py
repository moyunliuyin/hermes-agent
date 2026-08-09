"""Telegram DM rescue-phrase verification.

The phrase itself never lives in configuration. Operators store a scrypt hash
in ``GATEWAY_RESCUE_SECRET_HASH``; the gateway verifies it locally before
turning the plaintext message into the ``/rescue`` control command.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from functools import lru_cache


RESCUE_SECRET_HASH_ENV = "GATEWAY_RESCUE_SECRET_HASH"
RESCUE_PROFILE = "default"
RESCUE_TTL_SECONDS = 15 * 60

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_SALT_BYTES = 16
_SCRYPT_DKLEN = 32
_MAX_PHRASE_CHARS = 256


@lru_cache(maxsize=1)
def _default_profile_rescue_hash() -> str:
    """Load the process-wide verifier for an unscoped primary adapter task."""
    try:
        from agent.secret_scope import load_env_file
        from hermes_constants import get_hermes_home

        values = load_env_file(get_hermes_home() / ".env")
        return values.get(RESCUE_SECRET_HASH_ENV, "").strip()
    except Exception:
        return ""


def _rescue_secret_hash() -> str:
    """Read the verifier from the active profile's isolated secret scope."""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            return (get_secret(RESCUE_SECRET_HASH_ENV) or "").strip()
        except UnscopedSecretError:
            return _default_profile_rescue_hash()
    except ImportError:
        return os.getenv(RESCUE_SECRET_HASH_ENV, "").strip()


def rescue_phrase_configured() -> bool:
    """Return whether a rescue phrase hash is configured."""
    return bool(_rescue_secret_hash())


def verify_rescue_phrase(phrase: str) -> bool:
    """Verify *phrase* against the configured, bounded scrypt hash."""
    encoded = _rescue_secret_hash()
    if not encoded or not isinstance(phrase, str) or len(phrase) > _MAX_PHRASE_CHARS:
        return False

    try:
        scheme, n_s, r_s, p_s, salt_b64, expected_b64 = encoded.split("$")
        n, r, p = int(n_s), int(r_s), int(p_s)
        if (scheme, n, r, p) != ("scrypt", _SCRYPT_N, _SCRYPT_R, _SCRYPT_P):
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(expected_b64, validate=True)
        if len(salt) != _SCRYPT_SALT_BYTES or len(expected) != _SCRYPT_DKLEN:
            return False
        actual = hashlib.scrypt(
            phrase.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=0,
        )
    except (TypeError, ValueError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)
