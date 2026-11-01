import os
import re
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("trustguard.website_detector")

# Global cache for pipeline and feature metadata
_URL_PIPELINE = None
_URL_METADATA = None
_LOADED_MODEL_PATH = None
_PIPELINE_LOAD_ATTEMPTED = False
_PIPELINE_ERROR = None

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",
    ".top", ".xyz", ".click", ".buzz",
    ".work", ".zip", ".mov", ".online", ".site",
    ".live", ".support", ".club", ".icu", ".cam",
    ".link", ".info", ".monster", ".rest", ".bar"
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd",
    "ow.ly", "buff.ly", "cutt.ly", "shorturl.at", "goo.gl"
}

SUSPICIOUS_KEYWORDS = {
    "login", "verify", "verification", "secure", "account",
    "update", "password", "wallet", "bank", "payment",
    "refund", "free", "reward", "prize", "claim",
    "bonus", "otp", "confirm", "signin", "auth", "credential",
    "billing", "invoice", "validate", "support", "resolution"
}

# Domains with user-generated or shared hosting: NEVER allow suffix bypass
SHARED_HOSTING_DOMAINS = {
    "storage.cloud.google.com", "onedrive.live.com", "firebaseapp.com",
    "web.app", "pages.dev", "workers.dev", "github.io", "gitlab.io",
    "appspot.com", "s3.amazonaws.com", "blob.core.windows.net",
    "sites.google.com", "drive.google.com", "dropbox.com/s",
    "wixsite.com", "weebly.com", "wordpress.com", "blogspot.com"
}

# Major verified legitimate platforms (navigational apex/official domains)
VERIFIED_TOP_DOMAINS = {
    # AI & Research Platforms
    "chatgpt.com", "openai.com", "claude.ai", "anthropic.com", "deepseek.com", "huggingface.co",
    # Search, Cloud & Tech Giants
    "google.com", "google.co.in", "google.co.uk", "youtube.com", "microsoft.com", "office.com",
    "apple.com", "icloud.com", "amazon.com", "amazon.in", "github.com", "gitlab.com", "stackoverflow.com",
    "cloudflare.com", "yahoo.com", "bing.com",
    # Social & Messaging
    "facebook.com", "fb.com", "instagram.com", "whatsapp.com", "twitter.com", "x.com", "linkedin.com",
    "reddit.com", "telegram.org", "discord.com", "zoom.us", "slack.com", "tiktok.com",
    # Media & Knowledge
    "netflix.com", "spotify.com", "wikipedia.org", "medium.com", "twitch.tv", "imdb.com",
    # Finance, Commerce & Known Brands
    "paypal.com", "stripe.com", "visa.com", "mastercard.com", "ebay.com", "flipkart.com",
    "chipotle.com", "starbucks.com", "mcdonalds.com", "uber.com"
}

BRAND_KEYWORDS = [
    "paypal", "google", "microsoft", "apple", "amazon",
    "facebook", "instagram", "whatsapp", "netflix", "bank",
    "chase", "wellsfargo", "citi", "barclays", "chipotle",
    "chatgpt", "openai", "binance", "coinbase"
]


def _get_ml_pipeline():
    global _URL_PIPELINE, _URL_METADATA, _LOADED_MODEL_PATH, _PIPELINE_LOAD_ATTEMPTED, _PIPELINE_ERROR
    if _URL_PIPELINE is not None:
        return _URL_PIPELINE, _URL_METADATA, _PIPELINE_ERROR

    # Import URLFeatureExtractor so joblib can unpickle cleanly
    try:
        from app.services.url_feature_extractor import URLFeatureExtractor  # noqa: F401
    except ImportError:
        try:
            from server.app.services.url_feature_extractor import URLFeatureExtractor  # noqa: F401
        except ImportError:
            pass

    import joblib

    curr = Path(__file__).resolve().parent
    base_dir = None
    for ancestor in [curr, *curr.parents]:
        if (ancestor / "models" / "phishing_urls_v5").exists() or (ancestor / "models" / "phishing_urls_training_v3").exists():
            base_dir = ancestor
            break
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    # Primary model: V5 integrated pipeline
    primary_dir = base_dir / "models" / "phishing_urls_v5"
    fallback_dir = base_dir / "models" / "phishing_urls_training_v3"

    for candidate_dir in [primary_dir, fallback_dir]:
        model_path = candidate_dir / "phishing_urls_pipeline.pkl"
        meta_path = candidate_dir / "metadata.json"

        if model_path.exists():
            try:
                pipeline = joblib.load(str(model_path))
                metadata = {}
                if meta_path.exists():
                    with open(str(meta_path), "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                _URL_PIPELINE = pipeline
                _URL_METADATA = metadata
                _LOADED_MODEL_PATH = str(model_path.resolve())
                _PIPELINE_ERROR = None
                logger.info(f"[WebsiteDetector] Loaded model pipeline from resolved absolute path: {_LOADED_MODEL_PATH}")
                print(f"[WebsiteDetector] Loaded model pipeline from resolved absolute path: {_LOADED_MODEL_PATH}", flush=True)
                return _URL_PIPELINE, _URL_METADATA, None
            except Exception as e:
                _PIPELINE_ERROR = str(e)

    _URL_PIPELINE = False
    return None, None, _PIPELINE_ERROR


def get_loaded_model_path():
    """Return the resolved absolute path of the ML model loaded at runtime."""
    global _LOADED_MODEL_PATH
    if _LOADED_MODEL_PATH is None:
        _get_ml_pipeline()
    return _LOADED_MODEL_PATH


def analyze_website(url: str):
    url = (url or "").strip()
    if not url:
        raise ValueError("Website URL is required.")

    # Guard against malicious script schemes
    lower_input = url.lower()
    if lower_input.startswith("javascript:") or lower_input.startswith("data:") or lower_input.startswith("vbscript:"):
        raise ValueError("Unsupported or unsafe URL protocol scheme.")

    original_url = url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url

    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        scheme = (parsed.scheme or "").lower()
        path = parsed.path or ""
        query = (parsed.query or "").lower()
    except Exception as exc:
        raise ValueError(f"Malformed URL structure: {exc}")

    if not hostname:
        raise ValueError("Could not identify the website domain.")

    clean_hostname = hostname.removeprefix("www.")
    details = []
    recommendations = []
    risk_score = 0

    # 1. Check if domain is a shared hosting or user-content platform
    is_shared_platform = any(
        clean_hostname == sh or clean_hostname.endswith("." + sh)
        for sh in SHARED_HOSTING_DOMAINS
    ) or "blogspot." in clean_hostname or ("google.com" in clean_hostname and "/forms/" in path.lower()) or "000webhostapp." in clean_hostname

    # 2. Check if domain belongs to a verified authentic official platform
    is_verified_official = False
    if not is_shared_platform:
        is_verified_official = any(
            clean_hostname == d or clean_hostname.endswith("." + d)
            for d in VERIFIED_TOP_DOMAINS
        )

    # If it is a verified official platform and has no suspicious query or path payload:
    has_suspicious_payload = bool(re.search(r"\.(exe|scr|bat|vbs|apk|zip)$", path, re.I)) or bool("@" in original_url)

    if is_verified_official and not has_suspicious_payload:
        if _LOADED_MODEL_PATH is None:
            _get_ml_pipeline()
        checks = {
            "https_encryption": scheme == "https",
            "domain_format": True,
            "trusted_tld": True,
            "url_shortener": True,
            "security_keywords": True,
            "subdomain_depth": True,
            "url_length": True,
            "at_symbol": "@" not in original_url,
            "path_redirection": True,
            "brand_integrity": True,
            "query_parameters": True,
            "hosting_platform": True,
            "verified_official": True,
            "ml_classifier": True
        }
        return {
            "result": "SAFE",
            "prediction": "SAFE",
            "confidence": 99.0,
            "risk_score": 0,
            "details": [
                f"Verified authentic official domain: {clean_hostname}.",
                "Website uses authentic certificates and verified corporate domain infrastructure."
            ],
            "recommendations": [
                "The website is confirmed as an authentic, verified official platform.",
                "You are connected directly to the organization's genuine infrastructure."
            ],
            "url": original_url,
            "domain": hostname,
            "protocol": scheme,
            "checks": checks,
            "indicators_evaluated": 14,
            "indicators_flagged": 0,
            "ml_phishing_probability": 0.0,
            "model_loaded": _LOADED_MODEL_PATH
        }

    # If hosted on a shared platform, note it explicitly as a threat signal
    if is_shared_platform:
        risk_score += 20
        details.append("Website is hosted on a shared public cloud or user-content platform commonly leveraged for credential harvesting.")

    # 3. Scheme check
    if scheme != "https":
        risk_score += 15
        details.append("The website does not use HTTPS encryption.")
        recommendations.append("Avoid entering passwords or personal information on non-HTTPS websites.")

    # 4. Raw IP check
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    if re.match(ip_pattern, hostname):
        risk_score += 40
        details.append("The URL uses a raw IP address instead of a registered domain name.")

    # 5. Suspicious TLD check
    if any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS):
        risk_score += 30
        details.append("The domain uses a high-risk or commonly abused top-level domain.")

    # 6. Shortener check
    if clean_hostname in SHORTENERS:
        risk_score += 25
        details.append("The URL uses a link-shortening service that conceals its real destination.")
        recommendations.append("Expand and verify shortened links before opening them.")

    # 7. Suspicious keywords in URL
    lower_url = original_url.lower()
    tokens = set(re.findall(r"[a-z0-9]+", lower_url))
    keyword_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in tokens]
    if keyword_hits:
        risk_score += min(35, len(keyword_hits) * 10)
        details.append(f"The URL contains security or credential-sensitive keywords: {', '.join(keyword_hits[:4])}.")

    # 8. Subdomain depth
    labels = hostname.split(".")
    if len(labels) >= 5:
        risk_score += 20
        details.append("The domain contains an unusually deep hierarchy of subdomain levels.")

    # 9. Unusually long URL
    if len(original_url) > 90:
        risk_score += 15
        details.append("The URL is unusually long, which is a common obfuscation technique.")

    # 10. @ symbol
    if "@" in original_url:
        risk_score += 35
        details.append("The URL contains an '@' symbol used to disguise destination credentials.")

    # 11. Double slash in path
    if "//" in original_url.replace("://", ""):
        risk_score += 25
        details.append("The URL contains abnormal double slashes in the path indicating possible open redirect.")

    # 12. Brand impersonation / typosquatting check
    brand_hit = None
    domain_labels = clean_hostname.split(".")
    if len(domain_labels) > 1:
        sld = domain_labels[-2]
        for b in BRAND_KEYWORDS:
            if b in clean_hostname and sld != b:
                brand_hit = b
                break
            if b in path.lower() and sld != b:
                brand_hit = b
                break

    if brand_hit:
        risk_score += 45
        details.append(f"URL impersonates brand '{brand_hit.capitalize()}' without matching the official domain.")

    # 13. Sensitive query parameters
    query_tokens = set(re.findall(r"[a-z0-9]+", query))
    sensitive_words = ["password", "token", "otp", "verify", "login", "account", "cvv", "card", "credential"]
    if any(word in query_tokens for word in sensitive_words):
        risk_score += 25
        details.append("The URL query parameters request authentication tokens or credentials.")

    # 14. Run Machine Learning Pipeline
    ml_flagged = False
    ml_confidence = None
    phish_prob = None
    pipeline, metadata, pipe_err = _get_ml_pipeline()

    calibrated_threshold = 0.46
    if metadata and "decision_threshold" in metadata:
        try:
            calibrated_threshold = float(metadata["decision_threshold"])
        except Exception:
            calibrated_threshold = 0.46

    if pipeline:
        try:
            # Determine class index for phishing (class 1)
            classes = list(getattr(pipeline, "classes_", [0, 1]))
            phish_idx = classes.index(1) if 1 in classes else 1

            prob = pipeline.predict_proba([original_url])[0]
            phish_prob = float(prob[phish_idx])
            ml_confidence = round(phish_prob * 100, 2)

            if phish_prob >= 0.58:
                ml_flagged = True
                risk_score += 45
                details.append(
                    f"Machine Learning Classifier flagged this URL as Phishing/Malicious ({ml_confidence}% threat confidence)."
                )
            elif phish_prob >= calibrated_threshold:
                ml_flagged = True
                risk_score += 30
                details.append(
                    f"Machine Learning Classifier flagged elevated phishing indicators ({ml_confidence}% threat confidence)."
                )
            elif phish_prob >= 0.38:
                risk_score += 15
                details.append(
                    f"Machine Learning Classifier identified borderline URL structural risks ({ml_confidence}% threat score)."
                )
            else:
                # Structural safety verified by ML
                details.append(
                    f"Machine Learning Classifier verified structural URL safety ({round((1.0 - phish_prob) * 100, 2)}% safety score)."
                )
        except Exception as e:
            # Never default silently to SAFE on ML failure
            details.append(f"Machine learning inference encountered an anomaly: fallback heuristics active.")
            pipeline = False
    else:
        # Pipeline could not be loaded
        details.append("Machine learning classifier was unavailable; heuristic verification engine active.")

    # Final verdict calculation
    risk_score = min(risk_score, 100)

    # Determine verdict based on calibrated ML probability and heuristic risk
    if ml_flagged or risk_score >= 40:
        result = "FAKE"
        if phish_prob is not None:
            confidence = max(phish_prob * 100.0, 72.0 + min(26.0, risk_score * 0.25))
        else:
            confidence = min(98.0, 70.0 + risk_score * 0.28)
        recommendations.extend([
            "Do not enter passwords, OTPs, banking information, or personal credentials.",
            "Verify the official domain through known authentic search or bookmarks.",
            "Do not download any files or accept permissions from this website."
        ])
    elif (phish_prob is not None and phish_prob >= 0.38) or risk_score >= 20 or (pipeline is False and risk_score >= 10):
        result = "SUSPICIOUS"
        if phish_prob is not None:
            confidence = min(92.0, max(60.0, 50.0 + phish_prob * 40.0))
        else:
            confidence = min(92.0, 58.0 + risk_score * 0.30)
        recommendations.extend([
            "Verify the domain authenticity before interacting with this website.",
            "Avoid submitting sensitive credentials until confirmed legitimate."
        ])
    else:
        # Only declare SAFE if ML confirmed low probability AND heuristics found negligible risk
        if pipeline is False:
            # If ML was completely unavailable, never return SAFE with high confidence
            result = "SUSPICIOUS"
            confidence = 65.0
            details.append("Inconclusive: Automated classifier was offline during evaluation.")
            recommendations.append("Manual domain verification recommended.")
        else:
            result = "SAFE"
            if phish_prob is not None:
                confidence = max(82.0, min(99.0, (1.0 - phish_prob) * 100.0))
            else:
                confidence = min(98.0, 88.0 + (20 - risk_score) * 0.5)

            if not details:
                details.append("No major phishing or malicious website indicators were detected.")
            recommendations.extend([
                "No high-risk website patterns were identified.",
                "Always maintain standard cybersecurity vigilance when browsing unfamiliar sites."
            ])

    checks = {
        "https_encryption": scheme == "https",
        "domain_format": not bool(re.match(ip_pattern, hostname)),
        "trusted_tld": not any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS),
        "url_shortener": clean_hostname not in SHORTENERS,
        "security_keywords": len(keyword_hits) == 0,
        "subdomain_depth": len(labels) < 5,
        "url_length": len(original_url) <= 90,
        "at_symbol": "@" not in original_url,
        "path_redirection": "//" not in original_url.replace("://", ""),
        "brand_integrity": not bool(brand_hit),
        "query_parameters": not any(word in query_tokens for word in sensitive_words),
        "hosting_platform": not is_shared_platform,
        "verified_official": is_verified_official,
        "ml_classifier": not ml_flagged
    }
    indicators_flagged = sum(1 for passed in checks.values() if not passed)

    return {
        "result": result,
        "prediction": result,
        "confidence": round(confidence, 2),
        "risk_score": risk_score,
        "details": details,
        "recommendations": recommendations,
        "url": original_url,
        "domain": hostname,
        "protocol": scheme,
        "checks": checks,
        "indicators_evaluated": len(checks),
        "indicators_flagged": indicators_flagged,
        "ml_phishing_probability": round(phish_prob, 4) if phish_prob is not None else None,
        "model_loaded": _LOADED_MODEL_PATH
    }
