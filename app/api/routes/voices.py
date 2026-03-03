"""Voice listing and selection endpoints."""

from fastapi import APIRouter

from app.config import settings
from app.tts import TTS, SPEAKERS
from app.api.schemas import VoicesResponse, ChooseVoiceRequest

router = APIRouter(prefix="/voices", tags=["voices"])


@router.get("", response_model=VoicesResponse)
def list_voices():
    return VoicesResponse(
        speakers=TTS.list_voices(),
        assistant_speaker=settings.assistant_speaker,
    )


@router.post("/choose")
def choose_voice(req: ChooseVoiceRequest):
    if req.speaker not in SPEAKERS:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Unknown speaker: {req.speaker}")
    settings.assistant_speaker = req.speaker
    return {"assistant_speaker": settings.assistant_speaker}
