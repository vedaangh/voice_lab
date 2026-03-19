"""Training endpoints."""

import uuid

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import TrainingStartRequest, JobResponse, JobStatusResponse

router = APIRouter(prefix="/training", tags=["training"])

_jobs: dict[str, object] = {}


@router.post("/start", response_model=JobResponse)
def start_training(req: TrainingStartRequest, request: Request):
    if not req.dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    job_id = uuid.uuid4().hex[:12]

    config_dict = req.model_dump()
    config_dict["job_id"] = job_id

    import json
    config_json = json.dumps(config_dict)

    worker = request.app.state.training_worker
    call = worker.run.spawn(config_json=config_json)
    _jobs[job_id] = call
    return JobResponse(job_id=job_id, status="started")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def training_status(job_id: str):
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
