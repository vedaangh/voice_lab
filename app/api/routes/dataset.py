"""Dataset generation endpoints."""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import DatasetGenerateRequest, JobResponse, JobStatusResponse
from app.config import settings

router = APIRouter(prefix="/dataset", tags=["dataset"])

_jobs: dict[str, object] = {}


@router.post("/generate", response_model=JobResponse)
def generate_dataset(req: DatasetGenerateRequest, request: Request):
    if not req.input_path.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="input_path must be a .jsonl file")

    input_path = str(Path(settings.data_volume_path) / "input" / req.input_path)
    job_id = uuid.uuid4().hex[:12]
    output_dir = str(Path(settings.data_volume_path) / "output" / job_id)

    worker = request.app.state.pipeline_worker
    call = worker.run.spawn(
        input_path=input_path,
        output_dir=output_dir,
        assistant_speaker=req.assistant_speaker,
    )
    _jobs[job_id] = call
    return JobResponse(job_id=job_id, status="started")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def dataset_status(job_id: str):
    call = _jobs.get(job_id)
    if call is None:
        return JobStatusResponse(job_id=job_id, status="not_found")
    try:
        call.get(timeout=0)
        return JobStatusResponse(job_id=job_id, status="complete")
    except TimeoutError:
        return JobStatusResponse(job_id=job_id, status="running")
    except Exception as e:
        return JobStatusResponse(job_id=job_id, status="failed", detail=str(e))
