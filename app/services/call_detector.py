"""
TrustGuard AI - Phone Call Scam & Spoofed Caller Detector
Analyzes caller ID for telecom spoofing/number plan validity,
evaluates transcript using Kaggle-trained ML Scam Call classifier,
and applies multi-vector fraud heuristics (banking, OTP, urgency, remote access).
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from app.services.ml_call_service import MLCallService

_ml_call_service = MLCallService()


def validate_caller_id(caller: str) -> Tuple[bool, Optional[str], int]:
    """
    Validates caller ID format, detects telecom spoofing, impossible prefixes,
    and invalid telecom numbering plan structures.
    
    Returns: (is_suspicious, reason_message, risk_points)
    """
    c = caller.strip()
    if not c:
        return False, None, 0

    # Extract digits and plus sign
    digits_with_plus = re.sub(r"[^\d+]", "", c)
    clean_digits = re.sub(r"\D", "", c)

    if not clean_digits:
        # Non-numeric caller ID (e.g. unknown, anonymous)
        if any(w in c.lower() for w in ["unknown", "private", "anonymous", "hidden", "restricted"]):
            return True, "Caller ID is hidden or anonymous, preventing identity verification.", 25
        return False, None, 0

    # 1. Repetitive or sequential dummy numbers
    if len(clean_digits) >= 7:
        if len(set(clean_digits)) == 1:
            return True, f"Fake/dummy phone number ({c}): All digits are identical, characteristic of fraudulent autodialers.", 55
        if clean_digits in ["1234567890", "0123456789", "9876543210"]:
            return True, f"Fake/dummy phone number ({c}): Sequential digits detected, typical of spoofed VoIP robocalls.", 55

    # 2. National / International Numbering Plan checks
    # Standard 10-digit format
    if len(clean_digits) == 10:
        # In Indian National Numbering Plan (ITU-T E.164), 10-digit mobile numbers MUST start with 6, 7, 8, or 9.
        # In North America (NANPA), area codes cannot begin with 0 or 1.
        # Therefore, any 10-digit number starting with 0 or 1 is INVALID and SPOOFED.
        if clean_digits[0] in ["0", "1"]:
            return (
                True,
                f"Invalid Caller ID: Phone number '{c}' starts with '{clean_digits[0]}'. "
                f"Under telecommunications numbering standards (ITU-T & NANPA), genuine phone numbers cannot begin with 0 or 1. "
                f"This strongly indicates VoIP caller ID spoofing or fraudulent autodialer traffic.",
                50,
            )

    # 11-digit numbers without '+'
    elif len(clean_digits) == 11:
        # If starts with '1' (NANPA country code), the subsequent 3-digit area code CANNOT start with 0 or 1.
        # Example: '10045868901' -> area code prefix '00', which is illegal in NANPA.
        if clean_digits[0] == "1":
            area_start = clean_digits[1]
            if area_start in ["0", "1"]:
                return (
                    True,
                    f"Invalid Caller ID: Phone number '{c}' starts with invalid prefix '{clean_digits[:3]}'. "
                    f"Under telecommunications standards, no valid area code begins with '{area_start}'. "
                    f"This number is unallocated and represents spoofed robocall/VoIP traffic.",
                    50,
                )
        elif clean_digits[0] == "0":
            # 11-digit number starting with 0 (e.g. 0100...)
            return (
                True,
                f"Invalid Caller ID: Phone number '{c}' starts with illegal sequence '{clean_digits[:2]}', indicating spoofed telecommunications routing.",
                50,
            )

    # Short numbers or unallocated prefixes
    if clean_digits.startswith("100") or clean_digits.startswith("000") or clean_digits.startswith("111"):
        return (
            True,
            f"Suspicious Caller ID: Number '{c}' begins with reserved/emergency/unallocated prefix '{clean_digits[:3]}', typical of caller ID spoofing software.",
            45,
        )

    return False, None, 0


def analyze_call(caller: str, transcript: str) -> Dict[str, Any]:
    caller = (caller or "").strip()
    transcript = (transcript or "").strip()

    if not caller and not transcript:
        raise ValueError("Caller information or call transcript is required.")

    text = f"{caller} {transcript}".lower()
    details: List[str] = []
    risk_score = 0

    checks: Dict[str, Any] = {
        "caller_id_validation": "clear",
        "spoofed_number_risk": "clear",
        "ml_scam_model": "clear",
        "banking_impersonation": "clear",
        "credential_harvesting": "clear",
        "financial_urgency": "clear",
        "remote_access_threat": "clear",
        "identity_kyc_request": "clear",
    }

    # 1. Caller ID Telecom & Spoofing Validation
    is_spoofed, spoof_reason, spoof_risk = validate_caller_id(caller)
    if is_spoofed:
        risk_score += spoof_risk
        details.append(spoof_reason)
        checks["caller_id_validation"] = "flagged"
        checks["spoofed_number_risk"] = "flagged"

    # 2. Machine Learning Scam Call Classification
    ml_result = _ml_call_service.analyze(transcript if transcript else caller)
    ml_confidence = ml_result.get("confidence", 0.0)
    ml_scam_prob = ml_result.get("scam_probability", 0.0)
    is_ml_scam = ml_result.get("is_scam", False)

    if is_ml_scam and ml_scam_prob >= 0.60:
        ml_risk = int(min(45, ml_scam_prob * 45))
        risk_score += ml_risk
        checks["ml_scam_model"] = "flagged"
        details.append(
            f"Machine Learning Scam Model (trained on Kaggle fraud benchmark) classified this call as SCAM ({ml_confidence:.1f}% confidence)."
        )

    # 3. OTP / password / PIN / CVV credential requests
    credential_patterns = [
        r"\botp\b",
        r"one[\s-]?time password",
        r"\bpin\b",
        r"\bcvv\b",
        r"\bpassword\b",
        r"verification code",
        r"tell me.*otp",
        r"share.*otp",
        r"read.*otp",
    ]
    if any(re.search(pattern, text) for pattern in credential_patterns):
        risk_score += 35
        checks["credential_harvesting"] = "flagged"
        details.append("The caller requests sensitive credentials (OTP, PIN, CVV, or verification code).")

    # 4. Banking / payment / financial keywords
    banking_patterns = [
        r"\bbank\b",
        r"\bbanking\b",
        r"bank account",
        r"credit card",
        r"debit card",
        r"\bupi\b",
        r"payment",
        r"transfer money",
        r"send money",
        r"refund",
        r"transaction",
        r"account number",
        r"\bsbi\b",
        r"\bhdfc\b",
        r"\bicici\b",
    ]
    if any(re.search(pattern, text) for pattern in banking_patterns):
        checks["banking_impersonation"] = "flagged"
        if is_spoofed or is_ml_scam:
            risk_score += 30
            details.append("The caller claims banking or financial affiliation from an unverified or spoofed number.")
        else:
            risk_score += 15
            details.append("The conversation references banking, payments, or financial transactions.")

    # 5. Urgency / threats
    urgency_patterns = [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bact now\b",
        r"\bright now\b",
        r"account.*blocked",
        r"account.*suspended",
        r"account.*closed",
        r"\bpolice\b",
        r"legal action",
        r"court case",
        r"\barrest\b",
    ]
    if any(re.search(pattern, text) for pattern in urgency_patterns):
        risk_score += 25
        checks["financial_urgency"] = "flagged"
        details.append("The caller employs urgent, intimidating, or fear-based pressure tactics.")

    # 6. Remote access
    remote_patterns = [
        r"install.*app",
        r"download.*app",
        r"remote access",
        r"screen sharing",
        r"share your screen",
        r"\banydesk\b",
        r"\bteamviewer\b",
        r"remote desktop",
    ]
    if any(re.search(pattern, text) for pattern in remote_patterns):
        risk_score += 30
        checks["remote_access_threat"] = "flagged"
        details.append("The caller attempts to persuade the recipient to install remote access tools.")

    # 7. KYC / Identity requests
    identity_patterns = [
        r"\bkyc\b",
        r"aadhaar",
        r"pan card",
        r"identity proof",
        r"send.*document",
        r"share.*document",
        r"verify your identity",
    ]
    if any(re.search(pattern, text) for pattern in identity_patterns):
        risk_score += 20
        checks["identity_kyc_request"] = "flagged"
        details.append("The caller requests personal identity documents or KYC re-verification.")

    # Brief transcript warning
    if len(transcript.split()) <= 2 and is_spoofed:
        risk_score += 10
        details.append("High risk: Extremely abbreviated claim from an invalid caller ID.")

    risk_score = min(risk_score, 100)

    # Determine Verdict & Confidence
    if risk_score >= 50 or (is_spoofed and (is_ml_scam or "flagged" in checks.values())):
        result = "FAKE"  # or SCAM
        confidence = min(98.5, max(75.0, 65.0 + risk_score * 0.32))
    elif risk_score >= 25:
        result = "SUSPICIOUS"
        confidence = min(92.0, max(60.0, 55.0 + risk_score * 0.35))
    else:
        result = "SAFE"
        confidence = min(98.0, 92.0 + (25 - risk_score) * 0.24)

    if not details:
        details.append("No telecom spoofing or scam indicators detected. Number and transcript appear normal.")

    if result == "FAKE":
        recommendations = [
            "Do not answer or immediately terminate the call.",
            "Never share OTPs, passwords, PINs, or banking credentials.",
            "Do not install applications (AnyDesk, TeamViewer) at the caller's request.",
            "Block and report this number to the national telecom fraud portal (e.g., Chakshu / FTC).",
        ]
    elif result == "SUSPICIOUS":
        recommendations = [
            "Verify caller identity independently through the organization's official website.",
            "Do not disclose confidential details or transfer funds.",
            "Hang up and call back using the verified customer care number.",
        ]
    else:
        recommendations = [
            "No active scam indicators were detected.",
            "Maintain caution: Never provide one-time passwords (OTPs) over unexpected incoming calls.",
        ]

    return {
        "result": result,
        "prediction": result,
        "confidence": round(confidence, 2),
        "risk_score": risk_score,
        "details": details,
        "checks": checks,
        "analysis": checks,
        "recommendations": recommendations,
        "caller": caller,
        "transcript": transcript,
        "is_spoofed": is_spoofed,
        "ml_model_result": ml_result,
    }
