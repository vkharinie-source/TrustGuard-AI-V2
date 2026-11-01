from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.call_detector import analyze_call
from app.services.history_service import save_history


router = APIRouter(
    prefix="/api/analyze",
    tags=["Call Analysis"]
)


class CallRequest(BaseModel):
    caller: str = ""
    transcript: str = ""


@router.post("/call")
async def analyze_call_route(data: CallRequest):

    if not data.caller.strip() and not data.transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="Caller information or call transcript is required."
        )

    try:
        result = analyze_call(
            caller=data.caller,
            transcript=data.transcript
        )

        save_history(
            filename=data.caller[:80] or data.transcript[:80],
            prediction=result["result"],
            confidence=result["confidence"]
        )

        return {
            "success": True,
            **result
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
