"""P05 wire models; generated TypeScript consumes these same definitions."""
from typing import Annotated
from datetime import datetime
from pydantic import Field, field_validator
from .models import WireModel

class LoginInput(WireModel):
    username: Annotated[str, Field(min_length=3, max_length=64, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$')]
    password: Annotated[str, Field(min_length=1, max_length=1024, repr=False)]

class UserView(WireModel):
    id: str
    username: str

class SessionView(WireModel):
    user: UserView
    csrf_token: str
    expires_at: datetime

class Challenge(WireModel):
    csrf_token: str

class ProjectInput(WireModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]

    @field_validator('name')
    @classmethod
    def trim_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('A project name is required')
        return value

class ProjectView(WireModel):
    id: str
    name: str
    created_at: datetime

class ProjectList(WireModel):
    projects: list[ProjectView]
