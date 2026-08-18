"""Password hashing and token handling."""
import uuid

import pytest

from app.core.config import SecuritySettings
from app.core.exceptions import AuthenticationError
from app.core.security import (
    create_access_token,
    hash_password,
    subject_from_token,
    verify_password,
)

_CONFIG = SecuritySettings(
    secret_key="a-sufficiently-long-test-secret-key-value", bcrypt_rounds=4
)


def test_hash_is_salted_and_verifiable():
    first = hash_password("correct horse", _CONFIG)
    second = hash_password("correct horse", _CONFIG)

    assert first != second, "hashes are not salted"
    assert verify_password("correct horse", first)
    assert not verify_password("wrong horse", first)


def test_oversized_password_is_rejected():
    # bcrypt silently ignores bytes past 72; rejecting is honest.
    with pytest.raises(ValueError):
        hash_password("x" * 100, _CONFIG)


def test_malformed_stored_hash_is_a_failed_login_not_a_crash():
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_token_round_trip():
    user_id = uuid.uuid4()
    token, expires_at = create_access_token(user_id, "admin", _CONFIG)

    assert subject_from_token(token, _CONFIG) == user_id
    assert expires_at.tzinfo is not None, "expiry must be timezone-aware"


def test_token_signed_with_another_key_is_rejected():
    token, _ = create_access_token(uuid.uuid4(), "user", _CONFIG)
    other = SecuritySettings(secret_key="a-completely-different-but-long-secret-key")

    with pytest.raises(AuthenticationError):
        subject_from_token(token, other)


def test_garbage_token_is_rejected():
    with pytest.raises(AuthenticationError):
        subject_from_token("not.a.jwt", _CONFIG)


def test_non_uuid_subject_is_rejected_as_auth_failure():
    """A validly signed token with a junk subject must fail at the auth boundary,
    not deeper down as an unhandled 500."""
    from jose import jwt

    token = jwt.encode(
        {"sub": "not-a-uuid", "role": "user"},
        _CONFIG.secret_key.get_secret_value(),
        algorithm=_CONFIG.algorithm,
    )
    with pytest.raises(AuthenticationError):
        subject_from_token(token, _CONFIG)
