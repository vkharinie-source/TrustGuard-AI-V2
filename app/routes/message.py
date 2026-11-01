from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.message_detector import analyze_message
from app.services.history_service import save_history


router = APIRouter(
    prefix="/api/analyze",
    tags=["Message Analysis"]
)


class MessageRequest(BaseModel):
    content: str


@router.post("/message")
async def analyze_message_route(data: MessageRequest):

    if not data.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Message content is required."
        )

    try:
        result = analyze_message(data.content)

        save_history(
            filename=data.content[:80],
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
