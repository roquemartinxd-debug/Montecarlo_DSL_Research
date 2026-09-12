from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


_MAX_SEED = (1 << 128) - 1


class CreateJobRequest(BaseModel):
    source: str = Field(min_length=1, max_length=65_536)
    seed: str | None = None

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or not value.isascii() or not value.isdigit():
            raise ValueError("seed must be a non-negative decimal integer")
        number = int(value)
        if number > _MAX_SEED:
            raise ValueError("seed must fit in 128 bits")
        return str(number)


class DiagnosticPayload(BaseModel):
    code: str
    phase: str
    message: str
    line: int
    column: int
    start_offset: int
    end_offset: int
    lexeme: str | None = None
    expected: list[str] = Field(default_factory=list)
    found: str | None = None


class CompilationFailurePayload(BaseModel):
    error: Literal["compilation_failed"] = "compilation_failed"
    diagnostics: list[DiagnosticPayload]


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
    seed: str
    iterations: int
    websocket_path: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    seed: str
    iterations: int
    terminal: bool
    last_event: dict | None = None
