import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.history_service import save_history

try:
    from app.services.ml_news_service import MLNewsService
    _news_ml_service = MLNewsService()
except Exception:
    _news_ml_service = None

router = APIRouter(
    prefix="/api/analyze",
    tags=["News Analysis"]
)


class NewsRequest(BaseModel):
    headline: str = ""
    content: str = ""
    text: str = ""


@router.post("/news")
async def analyze_news_route(data: NewsRequest):
    """
    Inspect news articles, viral claims, and printed/scanned newspaper excerpts
    using a hybrid ML classification pipeline and heuristic sensationalism screening.
    """
    headline = (data.headline or "").strip()
    article_body = (data.content or data.text or "").strip()

    if not headline and not article_body:
        raise HTTPException(
            status_code=400,
            detail="Please provide news headline or article content to analyze."
        )

    combined_text = f"{headline}. {article_body}".strip() if headline else article_body
    lower_text = combined_text.lower()

    risk_score = 0
    details = []
    recommendations = []

    # 1. Machine Learning Fake News Classification
    ml_result = None
    if _news_ml_service and (_news_ml_service.pipeline or _news_ml_service.model):
        try:
            res = _news_ml_service.analyze(combined_text)
            ml_result = res.get("result")
            ml_conf = res.get("confidence", 50.0)

            if ml_result == "FAKE":
                risk_score += 45
                details.append(
                    f"Machine learning fake news classifier flagged this article as deceptive/unreliable ({ml_conf:.1f}% confidence)."
                )
            else:
                details.append(
                    f"Machine learning model found language patterns consistent with standard reporting ({ml_conf:.1f}% confidence)."
                )
        except Exception:
            pass

    # 2. Clickbait and Sensationalist Urgency Heuristics
    clickbait_patterns = [
        r"\bshocking\b",
        r"\bwon'?t believe\b",
        r"\bsecret revealed\b",
        r"\bthey don'?t want you to know\b",
        r"\bmiracle cure\b",
        r"\b100% guaranteed\b",
        r"\bdoctors hate (this|him|her)\b",
        r"\bshare before (it'?s|this is) deleted\b",
        r"\bbanned video\b",
        r"\bviral leak\b",
        r"\binstant riches\b",
    ]
    matched_clickbait = [p for p in clickbait_patterns if re.search(p, lower_text)]
    if matched_clickbait:
        risk_score += min(30, len(matched_clickbait) * 15)
        details.append("Article contains sensationalist clickbait phrasing or manipulative virality triggers.")

    # 3. Excessive All-Caps / Punctuation in Headline
    if headline:
        caps_ratio = sum(1 for c in headline if c.isupper()) / max(1, len(headline))
        if caps_ratio > 0.45 and len(headline) > 15:
            risk_score += 15
            details.append("Headline uses excessive uppercase lettering typical of unverified viral forwards.")
        if "!!!" in headline or "???" in headline or "?!" in headline:
            risk_score += 10
            details.append("Headline contains multiple exclamation or question marks typical of tabloid claims.")

    # 4. Anonymous Sourcing & Conspiracy Patterns
    conspiracy_patterns = [
        r"\bdeep state\b",
        r"\bunnamed whistleblower\b",
        r"\bhidden truth\b",
        r"\bmainstream media won'?t\b",
        r"\bsecret plot\b",
    ]
    if any(re.search(p, lower_text) for p in conspiracy_patterns):
        risk_score += 20
        details.append("Article relies on conspiracy narratives or unverifiable anonymous whistleblowers.")

    # 5. Verdict synthesis
    if risk_score >= 50 or ml_result == "FAKE":
        verdict = "FAKE"
        confidence = min(96.0, max(75.0, 60.0 + risk_score * 0.4))
        recommendations = [
            "Verify this claim through established international news agencies (e.g. AP, Reuters, BBC).",
            "Check official government, health, or institutional press releases before sharing.",
            "Do not redistribute or forward unverified viral news claims on messaging platforms."
        ]
    elif risk_score >= 25:
        verdict = "SUSPICIOUS"
        confidence = min(88.0, 58.0 + risk_score * 0.3)
        recommendations = [
            "Article contains sensationalist or unverified wording; cross-reference with credible outlets.",
            "Inspect author credentials and publication masthead for journalistic accountability."
        ]
    else:
        verdict = "REAL"
        confidence = min(95.0, 85.0 + (25 - risk_score) * 0.4)
        if not details:
            details.append("No overt sensationalism or deceptive markers were identified in the text.")
        recommendations = [
            "Article language conforms to standard journalistic reporting style.",
            "Always corroborate extraordinary claims with multiple independent sources."
        ]

    display_title = headline or article_body[:80]
    result_data = {
        "success": True,
        "result": verdict,
        "prediction": verdict,
        "confidence": round(float(confidence), 2),
        "risk_score": min(risk_score, 100),
        "details": details,
        "recommendations": recommendations,
        "headline": headline,
    }

    try:
        save_history(
            filename=display_title[:80],
            prediction=verdict,
            confidence=result_data["confidence"],
            category="News"
        )
    except Exception:
        pass

    return result_data
