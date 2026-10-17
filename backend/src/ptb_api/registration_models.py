"""Public account wire models. Tokens and passwords never appear in repr."""
from typing import Annotated
from pydantic import Field, field_validator
from .models import WireModel

Password = Annotated[str, Field(min_length=12, max_length=128, repr=False)]
Token = Annotated[str, Field(min_length=43, max_length=43, pattern=r'^[A-Za-z0-9_-]{43}$', repr=False)]


class EmailInput(WireModel):
    email: Annotated[str, Field(min_length=3, max_length=254, repr=False)]

    @field_validator('email')
    @classmethod
    def normalize_email(cls, value):
        import re
        value = value.strip().lower()
        # Explicit ASCII mailbox contract, not a permissive mail header parser.
        if not re.fullmatch(r"[a-z0-9!#$%&'*+/=?^_`{|}~.-]{1,64}@[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+", value):
            raise ValueError('Invalid email')
        local = value.split('@')[0]
        if local.startswith('.') or local.endswith('.') or '..' in local:
            raise ValueError('Invalid email')
        return value


class RegisterInput(EmailInput):
    username: Annotated[str, Field(min_length=3, max_length=64, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$')]
    password: Password


class VerifyEmailInput(WireModel):
    token: Token


class ResetPasswordInput(VerifyEmailInput):
    password: Password


class PublicAccountOptions(WireModel):
    registration: bool
    password_recovery: bool
    ephemeral_workspaces: bool = False


class EmailAccepted(WireModel):
    accepted: bool = True
