"""Pydantic request/response models."""

from pydantic import BaseModel


class DatasetGenerateRequest(BaseModel):
    input_path: str
    assistant_speaker: str | None = None


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None


class VoicesResponse(BaseModel):
    speakers: list[str]
    assistant_speaker: str


class ChooseVoiceRequest(BaseModel):
    speaker: str


