import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.website_detector import analyze_website as analyze_website_v5
from app.services.website_detector_v6_exp import analyze_website_v6
from app.services.history_service import save_history

router = APIRouter(
    prefix="/api/analyze",
    tags=["Website Analysis"]
)


def is_v6_enabled() -> bool:
    """Explicit staging/canary flag: defaults to False (retaining V5 production)."""
    val = os.getenv("ENABLE_WEBSITE_DETECTOR_V6", "false").strip().lower()
    return val in ("true", "1", "yes", "enabled", "on")


class WebsiteRequest(BaseModel):
    url: str


@router.post("/website")
async def analyze_website_route(data: WebsiteRequest):
    if not data.url.strip():
        raise HTTPException(
            status_code=400,
            detail="Website URL is required."
        )

    try:
        # Controlled canary routing: default to V5 production unless flag is explicitly enabled
        if is_v6_enabled():
            result = analyze_website_v6(data.url)
        else:
            result = analyze_website_v5(data.url)

        save_history(
            filename=data.url[:80],
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
