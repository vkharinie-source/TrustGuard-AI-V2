from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.routes.prediction import (
    router as prediction_router,
    handle_prediction,
)
from app.routes.auth import (
    router as auth_router,
    register as auth_register,
    login as auth_login,
    RegisterRequest,
    LoginRequest,
)
from app.routes.history import router as history_router
from app.routes.message import router as message_router
from app.routes.email import router as email_router
from app.routes.website import router as website_router
from app.routes.call import router as call_router
from app.routes.scams import router as scams_router
from app.routes.news import router as news_router

from app.services.history_service import get_history


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TrustGuard AI",
    description="AI-powered digital threat and fake-content detection platform",
    version="2.0.0",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",

        # Render production frontends & backends
        "https://trustguard-ai-v2-frontend.onrender.com",
        "https://trustguard-ai-v2-backend.onrender.com",
        "https://deepfake-detector-trustguard-ai-1.onrender.com",
        "https://deepfake-detector-trustguard-ai.onrender.com",
    ],
    # Automatically permit all onrender.com subdomains and local ports
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.onrender\.com$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(prediction_router)
app.include_router(auth_router)
app.include_router(history_router)
app.include_router(message_router)
app.include_router(email_router)
app.include_router(website_router)
app.include_router(call_router)
app.include_router(scams_router)
app.include_router(news_router)


# ============================================================
# COMPATIBILITY TOP-LEVEL ROUTE ALIASES
# Frontend compatibility across versions
# ============================================================

@app.post("/predict")
async def predict_top_level(file: UploadFile = File(...)):
    return await handle_prediction(file)


@app.post("/register")
@app.post("/auth/register")
async def register_top_level(data: RegisterRequest):
    return await auth_register(data)


@app.post("/login")
@app.post("/auth/login")
async def login_top_level(data: LoginRequest):
    return await auth_login(data)


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

@app.get("/api/dashboard/stats")
def dashboard_stats():
    items = get_history()
    total = len(items)

    safe_count = 0
    suspicious_count = 0
    threat_count = 0

    for item in items:
        pred = str(
            item.get("prediction", "")
            or item.get("result", "")
        ).upper()

        if pred in ("REAL", "SAFE", "LEGITIMATE"):
            safe_count += 1
        elif pred == "SUSPICIOUS":
            suspicious_count += 1
            threat_count += 1
        else:
            threat_count += 1

    return {
        "success": True,
        "stats": {
            "total": total,
            "safe": safe_count,
            "suspicious": suspicious_count,
            "threats_detected": threat_count,
        },
        "recent": items[:5],
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "message": "TrustGuard AI API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": [
            "POST /predict",
            "POST /api/predict/image",
            "POST /api/predict/video",
            "POST /api/predict/audio",
            "POST /api/forensics/ela",
            "POST /api/analyze/message",
            "POST /api/analyze/email",
            "POST /api/analyze/website",
            "POST /api/analyze/call",
            "POST /api/analyze/job",
            "POST /api/analyze/app",
            "POST /api/analyze/internship",
            "POST /api/analyze/news",
            "POST /api/auth/login",
            "POST /api/auth/register",
            "GET /api/history",
            "GET /api/dashboard/stats",
            "GET /api/transparency/stats",
            "GET /health",
        ],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "success": True,
        "status": "healthy",
        "model": "MobileNetV2",
        "version": "2.0.0",
    }
