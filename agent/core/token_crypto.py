"""Session-credential encryption for the lease-handover layer.

The Worker needs to act as the original user when it resumes a backgrounded
session — submitting HF Jobs, polling repos, etc. We don't want the plaintext
token in Mongo, so the session manager encrypts the user token with a Fernet
key held only in the ``SESSION_TOKEN_ENCRYPTION_KEY`` environment variable.
Compromise of Mongo alone is not sufficient to read tokens; the attacker
would also need the env key on either Main or Worker.

The function set is deliberately tiny — encrypt, decrypt, and a readiness
probe — so callers don't have to think about the crypto primitive. Missing
or malformed keys fail closed: ``encrypt`` returns ``None`` and ``decrypt``
returns ``None``, so the caller falls back to the "no token" path rather
than persisting plaintext or crashing the session-rebuild loop.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_ENV_KEY: Final[str] = "SESSION_TOKEN_ENCRYPTION_KEY"
_fernet: Fernet | None = None
_warned_missing_key: bool = False


def _get_fernet() -> Fernet | None:
    global _fernet, _warned_missing_key
    if _fernet is not None:
        return _fernet
    key = os.environ.get(_ENV_KEY, "").strip()
    if not key:
        if not _warned_missing_key:
            logger.warning(
                "%s not set — session credentials will NOT be persisted "
                "to Mongo; backgrounded sessions will require user "
                "reconnect to resume tool execution.",
                _ENV_KEY,
            )
            _warned_missing_key = True
        return None
    try:
        _fernet = Fernet(key.encode())
    except (ValueError, TypeError) as e:
        if not _warned_missing_key:
            logger.error(
                "%s is set but malformed (%s) — must be a 32-byte "
                "urlsafe-base64 string. Generate one with "
                "`python -c 'from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())'`.",
                _ENV_KEY, e,
            )
            _warned_missing_key = True
        return None
    return _fernet


def is_ready() -> bool:
    """True iff the encryption key is present and well-formed."""
    return _get_fernet() is not None


def encrypt(token: str | None) -> str | None:
    """Encrypt ``token`` to a Fernet ciphertext (urlsafe-base64 string).

    Returns ``None`` if the token is empty or the key isn't available —
    callers must treat ``None`` as "don't persist a blob, fall back to
    reconnect-driven flow on the Worker side".
    """
    if not token:
        return None
    f = _get_fernet()
    if f is None:
        return None
    return f.encrypt(token.encode()).decode()


def decrypt(ciphertext: str | None) -> str | None:
    """Decrypt a ciphertext produced by ``encrypt``.

    Returns ``None`` on any failure (missing key, malformed blob, tampered
    payload). The caller proceeds without a token rather than crashing.
    """
    if not ciphertext:
        return None
    f = _get_fernet()
    if f is None:
        return None
    try:
        return f.decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        logger.warning("decrypt: ciphertext rejected (key mismatch or tampering)")
        return None
    except Exception as e:
        logger.warning("decrypt: unexpected failure: %s", e)
        return None
