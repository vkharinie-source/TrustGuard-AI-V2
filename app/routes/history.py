from fastapi import APIRouter

from app.services.history_service import get_history, clear_history

router = APIRouter(tags=["History"])


@router.get("/history")
@router.get("/api/history")
def history():
    return {
        "success": True,
        "history": get_history(),
    }


@router.delete("/api/history")
@router.delete("/history")
def delete_history():
    clear_history()
    return {
        "success": True,
        "message": "History cleared successfully."
    }
