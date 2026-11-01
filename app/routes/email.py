from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.email_detector import analyze_email
from app.services.history_service import save_history


# ============================================================
# EMAIL ANALYSIS ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/analyze",
    tags=["Email Analysis"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class EmailRequest(BaseModel):
    subject: str = ""
    sender: str = ""
    content: str = ""


# ============================================================
# ANALYZE EMAIL
# ============================================================

@router.post("/email")
async def analyze_email_route(data: EmailRequest):
    """
    Analyze an email for suspicious or potentially fraudulent
    characteristics.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not (
        data.subject.strip()
        or data.sender.strip()
        or data.content.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="Please provide email subject, sender, or content.",
        )

    # --------------------------------------------------------
    # Run detector
    # --------------------------------------------------------

    try:
        result = analyze_email(
            subject=data.subject,
            sender=data.sender,
            content=data.content,
        )

        # ----------------------------------------------------
        # Save result to history
        # ----------------------------------------------------

        filename = (
            data.subject[:80]
            if data.subject.strip()
            else data.sender[:80]
        )

        save_history(
            filename=filename,
            prediction=result["result"],
            confidence=result["confidence"],
        )

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {
            "success": True,
            **result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Email analysis failed: {str(exc)}",
        ) from exc