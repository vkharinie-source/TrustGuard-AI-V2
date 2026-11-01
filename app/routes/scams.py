import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.history_service import save_history

try:
    from app.services.ml_job_scam_service import MLJobScamService
    job_scam_service = MLJobScamService()
except Exception:
    job_scam_service = None


router = APIRouter(
    prefix="/api/analyze",
    tags=["Specialized Analyzers"]
)


class JobRequest(BaseModel):
    title: str = ""
    company: str = ""
    description: str = ""


class InternshipRequest(BaseModel):
    title: str = ""
    company: str = ""
    description: str = ""


class AppRequest(BaseModel):
    app_name: str = ""
    developer: str = ""
    description: str = ""


def detect_job_or_internship_scam(title: str, company: str, description: str, category: str):
    combined = f"{title} {company} {description}".lower()
    details = []
    risk_score = 0

    fee_patterns = [
        r"\bregistration fee\b",
        r"\btraining fee\b",
        r"\bsecurity deposit\b",
        r"\bprocessing fee\b",
        r"\bpay upfront\b",
        r"\bpayment required\b",
        r"\bfee required\b",
        r"\bcharges apply\b",
        r"\bwire transfer\b",
        r"\bcrypto\b",
        r"\bbitcoin\b"
    ]
    if any(re.search(pat, combined) for pat in fee_patterns):
        risk_score += 45
        details.append(f"The {category.lower()} offer asks for upfront payment, registration fee, or security deposit.")

    suspicious_comm = [
        r"\btelegram\b",
        r"\bwhatsapp only\b",
        r"\bno interview\b",
        r"\binstant hiring\b",
        r"\bguaranteed placement\b"
    ]
    if any(re.search(pat, combined) for pat in suspicious_comm):
        risk_score += 30
        details.append(f"Unusual communication or guaranteed selection without rigorous evaluation.")

    salary_red_flags = [
        r"\b\$\s*\d{3,}\s*(per|/)\s*(day|hour)\b",
        r"\bearn\s+\$\s*\d{3,}\b",
        r"\bpart[\s-]?time.*daily payout\b",
        r"\bdata entry.*5000\b"
    ]
    if any(re.search(pat, combined) for pat in salary_red_flags):
        risk_score += 25
        details.append("Unrealistically high remuneration promises typical of recruitment fraud.")

    ml_verdict = None
    if job_scam_service and description.strip():
        try:
            ml_res = job_scam_service.analyze(description)
            if ml_res.get("result") in ("SCAM", "FAKE"):
                risk_score += 35
                details.append("Machine learning classifier flagged scam indicators in the job text.")
            ml_verdict = ml_res.get("result")
        except Exception:
            pass

    if risk_score >= 50 or ml_verdict in ("SCAM", "FAKE"):
        verdict = "SCAM"
        confidence = min(98.0, max(75.0, 60.0 + risk_score * 0.4))
        recommendations = [
            f"Never pay any upfront registration or training fee for a {category.lower()}.",
            "Verify the company directly via their official career portal and LinkedIn.",
            "Avoid sharing government IDs or bank account numbers until the employer is authenticated."
        ]
    elif risk_score >= 25:
        verdict = "SUSPICIOUS"
        confidence = min(90.0, 55.0 + risk_score * 0.3)
        recommendations = [
            f"Verify that the contact email matches the organization's verified domain.",
            "Search reviews and scam warnings regarding this recruiter before proceeding."
        ]
    else:
        verdict = "SAFE"
        confidence = min(96.0, 88.0 + (25 - risk_score) * 0.3)
        details.append(f"No overt {category.lower()} scam markers detected in the provided details.")
        recommendations = [
            f"The {category.lower()} appears standard based on the provided text.",
            "Always follow standard verification practices when accepting offers."
        ]

    return {
        "result": verdict,
        "prediction": verdict,
        "confidence": round(confidence, 2),
        "details": details,
        "recommendations": recommendations,
        "risk_score": min(risk_score, 100)
    }


@router.post("/job")
async def analyze_job_route(data: JobRequest):
    if not (data.title.strip() or data.company.strip() or data.description.strip()):
        raise HTTPException(status_code=400, detail="Job title, company, or description is required.")

    result = detect_job_or_internship_scam(data.title, data.company, data.description, "Job")
    display_name = data.title.strip() or data.company.strip() or data.description[:80]
    save_history(
        filename=display_name[:80],
        prediction=result["prediction"],
        confidence=result["confidence"],
        category="Job"
    )
    return {"success": True, **result}


@router.post("/internship")
async def analyze_internship_route(data: InternshipRequest):
    if not (data.title.strip() or data.company.strip() or data.description.strip()):
        raise HTTPException(status_code=400, detail="Internship title, company, or description is required.")

    result = detect_job_or_internship_scam(data.title, data.company, data.description, "Internship")
    display_name = data.title.strip() or data.company.strip() or data.description[:80]
    save_history(
        filename=display_name[:80],
        prediction=result["prediction"],
        confidence=result["confidence"],
        category="Internship"
    )
    return {"success": True, **result}


@router.post("/app")
async def analyze_app_route(data: AppRequest):
    if not (data.app_name.strip() or data.developer.strip() or data.description.strip()):
        raise HTTPException(status_code=400, detail="Application name, developer, or description is required.")

    combined = f"{data.app_name} {data.developer} {data.description}".lower()
    details = []
    risk_score = 0

    dangerous_perms = [
        r"\baccessibility service\b",
        r"\bread sms\b",
        r"\breceive sms\b",
        r"\bnotification listener\b",
        r"\bdevice administrator\b",
        r"\boverlay\b",
        r"\bdraw over other apps\b"
    ]
    if any(re.search(pat, combined) for pat in dangerous_perms):
        risk_score += 40
        details.append("App requires high-risk permissions commonly abused by banking trojans or spyware.")

    sideloading = [
        r"\bapk\b",
        r"\bunknown sources\b",
        r"\bdisable play protect\b",
        r"\bnot on play store\b",
        r"\bexternal download\b"
    ]
    if any(re.search(pat, combined) for pat in sideloading):
        risk_score += 35
        details.append("Promotion of third-party APK sideloading or disabling security protections.")

    fake_app_types = [
        r"\binstant loan\b",
        r"\b7 day loan\b",
        r"\bram booster\b",
        r"\bbattery saver 200%\b",
        r"\bfree recharge\b",
        r"\bhack wifi\b"
    ]
    if any(re.search(pat, combined) for pat in fake_app_types):
        risk_score += 35
        details.append("Characteristics match known predatory loan or deceptive utility app types.")

    if risk_score >= 50:
        verdict = "FAKE"
        confidence = min(97.0, 70.0 + risk_score * 0.3)
        recommendations = [
            "Do not install applications from outside official app stores.",
            "Never grant Accessibility or SMS permissions to untrusted applications.",
            "Uninstall and scan your device if this app was already installed."
        ]
    elif risk_score >= 25:
        verdict = "SUSPICIOUS"
        confidence = min(88.0, 58.0 + risk_score * 0.3)
        recommendations = [
            "Verify developer credibility and reviews before granting runtime permissions.",
            "Only download verified applications with strong reputations."
        ]
    else:
        verdict = "SAFE"
        confidence = min(96.0, 89.0 + (25 - risk_score) * 0.25)
        details.append("No abnormal or dangerous application indicators were identified.")
        recommendations = [
            "The application description does not present known high-risk red flags.",
            "Always inspect runtime permissions requested by mobile apps."
        ]

    display_name = data.app_name.strip() or data.developer.strip() or data.description[:80]
    result = {
        "result": verdict,
        "prediction": verdict,
        "confidence": round(confidence, 2),
        "details": details,
        "recommendations": recommendations,
        "risk_score": min(risk_score, 100)
    }
    save_history(
        filename=display_name[:80],
        prediction=verdict,
        confidence=result["confidence"],
        category="Application"
    )
    return {"success": True, **result}
