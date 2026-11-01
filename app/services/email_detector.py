import re
from app.services.ml_email_service import MLEmailService

_ml_email_service = MLEmailService()


def analyze_email(
    sender: str = "",
    subject: str = "",
    content: str = "",
) -> dict:
    """
    Hybrid email threat detector combining trained Kaggle ML classification
    with multi-factor heuristic checks (spoofed domains, urgency, credentials).

    Returns:
        result
        confidence
        risk_score
        details
        recommendations
        sender
        subject
        content
    """

    sender = (sender or "").strip()
    subject = (subject or "").strip()
    content = (content or "").strip()

    full_text = f"{sender} {subject} {content}".lower()

    risk_score = 0
    details = []
    recommendations = []

    # Verified legitimate corporate email domains
    verified_official_domains = {
        "paypal.com", "steampowered.com", "valvesoftware.com", "amazon.com", "amazon.in",
        "google.com", "apple.com", "microsoft.com", "github.com", "netflix.com",
        "spotify.com", "linkedin.com", "uber.com", "chase.com", "wellsfargo.com",
        "bankofamerica.com", "citi.com"
    }

    # Extract sender domain
    sender_domain = ""
    if "@" in sender:
        sender_domain = sender.split("@", 1)[1].lower().strip(">").strip()

    is_verified_sender = any(
        sender_domain == vd or sender_domain.endswith("." + vd)
        for vd in verified_official_domains
    )

    # Check for authentic transaction receipts / order confirmations
    receipt_patterns = [
        r"\byour receipt (?:from|for)\b",
        r"\border confirmation\b",
        r"\binvoice #?\d+\b",
        r"\byou sent a payment\b",
        r"\bpayment received\b",
        r"\bthank you for your (?:payment|order|purchase)\b",
    ]
    is_receipt = any(re.search(pat, full_text) for pat in receipt_patterns)

    # ========================================================
    # MACHINE LEARNING EMAIL CLASSIFICATION
    # ========================================================
    ml_res = _ml_email_service.analyze(full_text)
    is_ml_phishing = ml_res.get("is_phishing", False)
    ml_prob = ml_res.get("phish_probability", 0.0)
    ml_conf = ml_res.get("confidence", 0.0)

    # If email is a clean receipt from a verified sender, do not flag ML false positive
    if is_verified_sender and is_receipt:
        is_ml_phishing = False
        details.append(f"Authentic transaction receipt from verified organization domain ({sender_domain}).")

    if is_ml_phishing and ml_prob >= 0.70:
        ml_risk = int(min(45, ml_prob * 45))
        risk_score += ml_risk
        details.append(
            f"Machine Learning Classifier (Kaggle Phishing Benchmark) flagged this email as PHISHING ({ml_conf:.1f}% confidence)."
        )
    elif is_ml_phishing and ml_prob >= 0.50:
        risk_score += 20
        details.append(
            f"Machine Learning Classifier detected elevated phishing indicators ({ml_conf:.1f}% confidence)."
        )


    # ========================================================
    # SUSPICIOUS / FREE EMAIL DOMAIN
    # ========================================================

    suspicious_domains = [
        "tempmail",
        "10minutemail",
        "guerrillamail",
        "mailinator",
        "protonmail",
    ]

    sender_domain = ""

    if "@" in sender:
        sender_domain = sender.split("@", 1)[1].lower()

        if any(domain in sender_domain for domain in suspicious_domains):
            risk_score += 20
            details.append(
                "Sender uses a temporary, suspicious, or privacy-focused email domain."
            )
            recommendations.append(
                "Verify the sender through an independent official channel."
            )

        suspicious_tlds = [
            ".xyz",
            ".top",
            ".click",
            ".gq",
            ".tk",
            ".ml",
            ".cf",
            ".work",
        ]

        if any(sender_domain.endswith(tld) for tld in suspicious_tlds):
            risk_score += 20
            details.append(
                "Sender uses a domain with a potentially suspicious top-level domain."
            )
            recommendations.append(
                "Check whether the sender domain matches the organization's official domain."
            )

    # ========================================================
    # URGENCY / THREAT
    # ========================================================

    urgency_patterns = [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bact now\b",
        r"\blast warning\b",
        r"\baction required\b",
        r"\byour account will be blocked\b",
        r"\byour account will be suspended\b",
        r"\byour account will be closed\b",
        r"\bwithin \d+ minutes?\b",
        r"\bwithin \d+ hours?\b",
    ]

    if any(re.search(pattern, full_text) for pattern in urgency_patterns):
        risk_score += 20
        details.append(
            "Email uses urgency, threats, or pressure to make the recipient act quickly."
        )
        recommendations.append(
            "Do not act immediately. Verify the request independently."
        )

    # ========================================================
    # OTP / PASSWORD / CREDENTIALS
    # ========================================================

    credential_patterns = [
        r"\botp\b",
        r"\bone[- ]time password\b",
        r"\bpassword\b",
        r"\bpin\b",
        r"\bcvv\b",
        r"\bpasscode\b",
        r"\bverification code\b",
        r"\bsecurity code\b",
        r"\blogin credentials\b",
        r"\busername and password\b",
    ]

    if any(re.search(pattern, full_text) for pattern in credential_patterns):
        risk_score += 25
        details.append(
            "Email requests passwords, OTPs, PINs, CVV, or other sensitive credentials."
        )
        recommendations.append(
            "Never share OTPs, passwords, PINs, CVVs, or verification codes."
        )

    # ========================================================
    # BANKING / PAYMENT
    # ========================================================

    banking_patterns = [
        r"\bbank\b",
        r"\baccount\b",
        r"\bupi\b",
        r"\bpayment\b",
        r"\btransaction\b",
        r"\brefund\b",
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bnet banking\b",
        r"\bwallet\b",
        r"\btransfer\b",
    ]

    if any(re.search(pattern, full_text) for pattern in banking_patterns):
        risk_score += 15
        details.append(
            "Email contains banking, payment, transaction, or financial activity."
        )
        recommendations.append(
            "Verify financial requests using the organization's official website or application."
        )

    # ========================================================
    # REWARD / PRIZE / LOTTERY
    # ========================================================

    reward_patterns = [
        r"\byou won\b",
        r"\bcongratulations\b",
        r"\bprize\b",
        r"\breward\b",
        r"\blottery\b",
        r"\bcash prize\b",
        r"\bfree gift\b",
        r"\bclaim your reward\b",
    ]

    if any(re.search(pattern, full_text) for pattern in reward_patterns):
        risk_score += 15
        details.append(
            "Email contains unexpected prize, lottery, reward, or gift claims."
        )
        recommendations.append(
            "Do not pay money or submit personal information to claim an unexpected reward."
        )

    # ========================================================
    # SUSPICIOUS LINKS
    # ========================================================

    urls = re.findall(
        r"(https?://[^\s]+|www\.[^\s]+)",
        full_text,
    )

    if urls:
        shorteners = [
            "bit.ly",
            "tinyurl.com",
            "t.co",
            "goo.gl",
            "is.gd",
            "cutt.ly",
            "shorturl.at",
        ]

        suspicious_link_words = [
            "verify",
            "login",
            "secure",
            "account",
            "update",
            "confirm",
            "claim",
            "wallet",
            "bank",
            "password",
        ]

        suspicious_link = False

        for url in urls:
            if any(shortener in url for shortener in shorteners):
                suspicious_link = True

            if any(word in url for word in suspicious_link_words):
                suspicious_link = True

        if suspicious_link:
            risk_score += 20
            details.append(
                "Email contains a suspicious or potentially credential-related link."
            )
            recommendations.append(
                "Do not click suspicious links. Open the official website manually."
            )
        else:
            risk_score += 5
            details.append(
                "Email contains an external link."
            )

    # ========================================================
    # PERSONAL / SENSITIVE INFORMATION
    # ========================================================

    sensitive_patterns = [
        r"\baadhar\b",
        r"\bpan card\b",
        r"\bpassport\b",
        r"\bdate of birth\b",
        r"\baccount number\b",
        r"\bcard number\b",
        r"\bsocial security\b",
    ]

    if any(re.search(pattern, full_text) for pattern in sensitive_patterns):
        risk_score += 15
        details.append(
            "Email contains or requests sensitive personal information."
        )
        recommendations.append(
            "Do not send identity or financial information through an unverified email."
        )

    # ========================================================
    # ACCOUNT SECURITY / PHISHING LANGUAGE
    # ========================================================

    phishing_patterns = [
        r"\bverify your account\b",
        r"\bconfirm your account\b",
        r"\bupdate your account\b",
        r"\blog in to verify\b",
        r"\bsecurity alert\b",
        r"\bunusual activity\b",
        r"\bsuspicious activity\b",
        r"\baccount verification\b",
    ]

    if any(re.search(pattern, full_text) for pattern in phishing_patterns):
        risk_score += 15
        details.append(
            "Email contains common account-security or phishing language."
        )
        recommendations.append(
            "Go directly to the official website instead of using links in the email."
        )

    # ========================================================
    # CAP SCORE
    # ========================================================

    risk_score = min(risk_score, 100)

    # Verified domain receipt protection
    if is_verified_sender and is_receipt:
        risk_score = 0
        result = "REAL"
        confidence = 99.0
    elif risk_score >= 45 or (is_ml_phishing and risk_score >= 35):
        result = "FAKE"
        base_conf = ml_conf if is_ml_phishing else 70.0
        confidence = min(99.0, max(base_conf, 55.0 + (risk_score * 0.45)))
    elif risk_score >= 20 or is_ml_phishing:
        result = "SUSPICIOUS"
        confidence = min(92.0, 55.0 + (risk_score * 0.4))
    else:
        result = "REAL"
        confidence = min(99.0, 100.0 - (risk_score * 0.5))

    # ========================================================
    # DEFAULT RECOMMENDATION
    # ========================================================

    if not recommendations:
        recommendations.append(
            "No major suspicious patterns were detected. Still verify unexpected emails independently."
        )

    checks = {
        "ml_phishing_model": "flagged" if is_ml_phishing else "clear",
        "sender_domain_safety": "flagged" if "@" in sender and any(domain in sender_domain for domain in suspicious_domains) else "clear",
        "credential_harvesting": "flagged" if any(re.search(pattern, full_text) for pattern in phishing_patterns) else "clear",
        "urgency_intimidation": "flagged" if any(re.search(pattern, full_text) for pattern in urgency_patterns) else "clear",
    }

    return {
        "result": result,
        "prediction": result,
        "confidence": round(confidence, 2),
        "risk_score": risk_score,
        "details": details,
        "checks": checks,
        "analysis": checks,
        "recommendations": recommendations,
        "sender": sender,
        "subject": subject,
        "content": content,
    }