import ipaddress
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd


def extract_url_features(raw_url: str, feature_names: list[str]) -> pd.DataFrame:
    """Build a one-row feature frame in the saved model's exact column order."""
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("A non-empty URL is required.")

    value = raw_url.strip()
    if not re.match(r"^https?://", value, re.IGNORECASE):
        value = "https://" + value

    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Invalid URL: hostname could not be determined.")

    url = value.lower()
    path = parsed.path or ""
    host_without_www = hostname[4:] if hostname.startswith("www.") else hostname
    host_parts = [part for part in host_without_www.split(".") if part]
    words = re.findall(r"[a-zA-Z0-9]+", url)
    host_words = re.findall(r"[a-zA-Z0-9]+", host_without_www)
    path_words = re.findall(r"[a-zA-Z0-9]+", path)

    try:
        ipaddress.ip_address(hostname)
        uses_ip = 1
    except ValueError:
        uses_ip = 0

    digit_count = sum(ch.isdigit() for ch in url)
    host_digit_count = sum(ch.isdigit() for ch in hostname)
    repeated_chars = sum(max(0, n - 1) for n in Counter(url).values())
    shortening_domains = {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
        "is.gd", "buff.ly", "cutt.ly", "shorturl.at",
    }
    suffix = host_parts[-1] if host_parts else ""
    suspicious_tlds = {
        "zip", "mov", "top", "xyz", "click", "work", "gq", "tk", "ml", "cf"
    }

    values = {
        "length_url": len(url),
        "length_hostname": len(hostname),
        "ip": uses_ip,
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_or": url.count("|"),
        "nb_eq": url.count("="),
        "nb_underscore": url.count("_"),
        "nb_tilde": url.count("~"),
        "nb_percent": url.count("%"),
        "nb_slash": url.count("/"),
        "nb_star": url.count("*"),
        "nb_colon": url.count(":"),
        "nb_comma": url.count(","),
        "nb_semicolumn": url.count(";"),
        "nb_dollar": url.count("$"),
        "nb_space": url.count(" "),
        "nb_www": url.count("www"),
        "nb_com": url.count(".com"),
        "nb_dslash": url.count("//"),
        "http_in_path": int("http" in path.lower()),
        "https_token": int("https" in host_without_www),
        "ratio_digits_url": digit_count / max(1, len(url)),
        "ratio_digits_host": host_digit_count / max(1, len(hostname)),
        "punycode": int("xn--" in hostname),
        "port": int(parsed.port is not None),
        "tld_in_path": int(bool(suffix) and suffix in path.lower()),
        "tld_in_subdomain": int(len(host_parts) > 2 and suffix in ".".join(host_parts[:-2])),
        "abnormal_subdomain": int(len(host_parts) > 3),
        "nb_subdomains": max(0, len(host_parts) - 2),
        "prefix_suffix": int("-" in host_without_www),
        "random_domain": 0,
        "shortening_service": int(host_without_www in shortening_domains),
        "path_extension": int(bool(re.search(r"\.[a-zA-Z0-9]{1,8}$", path))),
        "nb_redirection": 0,
        "nb_external_redirection": 0,
        "length_words_raw": len(words),
        "char_repeat": repeated_chars,
        "shortest_words_raw": min(map(len, words), default=0),
        "shortest_word_host": min(map(len, host_words), default=0),
        "shortest_word_path": min(map(len, path_words), default=0),
        "longest_words_raw": max(map(len, words), default=0),
        "longest_word_host": max(map(len, host_words), default=0),
        "longest_word_path": max(map(len, path_words), default=0),
        "avg_words_raw": sum(map(len, words)) / max(1, len(words)),
        "avg_word_host": sum(map(len, host_words)) / max(1, len(host_words)),
        "avg_word_path": sum(map(len, path_words)) / max(1, len(path_words)),
        "phish_hints": sum(token in url for token in (
            "login", "verify", "update", "secure", "account", "password",
            "signin", "confirm", "bank", "wallet",
        )),
        "domain_in_brand": 0,
        "brand_in_subdomain": 0,
        "brand_in_path": 0,
        "suspecious_tld": int(suffix in suspicious_tlds),
        "statistical_report": 0,
    }

    # These features require fetching/analyzing the live webpage or external
    # reputation/DNS/registration data. Do not invent values for them.
    unavailable_features = {
        "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks",
        "ratio_nullHyperlinks", "nb_extCSS", "ratio_intRedirection",
        "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors",
        "login_form", "external_favicon", "links_in_tags", "submit_email",
        "ratio_intMedia", "ratio_extMedia", "sfh", "iframe", "popup_window",
        "safe_anchor", "onmouseover", "right_clic", "empty_title",
        "domain_in_title", "domain_with_copyright", "whois_registered_domain",
        "domain_registration_length", "domain_age", "web_traffic",
        "dns_record", "google_index", "page_rank",
    }

    missing = [name for name in feature_names
               if name not in values and name not in unavailable_features]
    if missing:
        raise ValueError(f"Feature extraction is not implemented for: {missing}")

    row = {
        name: values[name] if name in values else float("nan")
        for name in feature_names
    }
    row["url"] = value
    cols = (["url"] if "url" not in feature_names else []) + list(feature_names)
    return pd.DataFrame([row])[cols]


import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SUSPICIOUS_TLDS_SET = {
    "xyz", "top", "click", "buzz", "work", "gq", "tk", "ml", "cf",
    "online", "site", "live", "support", "club", "icu", "cam", "zip", "mov", "ru",
    "link", "info", "monster", "rest", "bar"
}

SHORTENERS_SET = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly", "cutt.ly", "shorturl.at", "goo.gl"
}

SECURITY_WORDS_SET = {
    "login", "verify", "verification", "secure", "account", "update", "password",
    "wallet", "bank", "payment", "refund", "free", "reward", "prize", "claim",
    "bonus", "otp", "confirm", "signin", "auth", "credential", "recover", "security",
    "billing", "invoice", "validate", "support", "service", "customer", "resolution",
    "identity", "unlock", "access", "portal"
}

BRANDS_LIST = [
    "paypal", "google", "microsoft", "apple", "amazon", "facebook", "instagram",
    "whatsapp", "netflix", "chase", "wellsfargo", "citi", "bankofamerica",
    "binance", "coinbase", "steamcommunity", "roblox", "ebay", "walmart", "adobe",
    "yahoo", "dropbox", "onedrive", "outlook", "icloud"
]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Self-contained transformer extracting 32 deterministic features from raw URL strings.
    Safe for production: no network lookups, purely lexical and structural parsing.
    """
    def __init__(self):
        self.feature_names_ = [
            "url_len", "host_len", "path_len", "query_len", "is_https", "is_ip",
            "dot_count", "subdomain_count", "hyphen_count", "host_hyphen_count",
            "at_count", "qm_count", "and_count", "eq_count", "slash_count",
            "percent_count", "double_slash_in_path", "suspicious_tld", "is_shortener",
            "digit_ratio_url", "digit_ratio_host", "host_entropy", "url_entropy",
            "phish_keyword_count", "brand_impersonation_flag", "http_in_path",
            "punycode", "word_count", "longest_word", "avg_word_len",
            "has_port", "tld_in_subdomain_or_path"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        features = []
        for raw in X:
            u = str(raw).strip()
            if not u.startswith("http://") and not u.startswith("https://"):
                full_u = "https://" + u
            else:
                full_u = u

            try:
                parsed = urlparse(full_u)
                host = (parsed.hostname or "").lower()
                path = parsed.path or ""
                query = parsed.query or ""
                scheme = (parsed.scheme or "").lower()
                port = parsed.port
            except Exception:
                host, path, query, scheme, port = "", "", "", "", None

            clean_host = host.removeprefix("www.")
            host_parts = [p for p in clean_host.split(".") if p]
            tld = host_parts[-1] if host_parts else ""

            try:
                ipaddress.ip_address(host)
                is_ip = 1.0
            except ValueError:
                is_ip = 0.0

            words = re.findall(r"[a-z0-9]+", u.lower())

            sld = host_parts[-2] if len(host_parts) >= 2 else (host_parts[0] if host_parts else "")
            brand_flag = 0.0
            for b in BRANDS_LIST:
                if b in clean_host and sld != b:
                    brand_flag = 1.0
                    break
                if b in path.lower() and sld != b:
                    brand_flag = 1.0
                    break

            phish_hits = sum(1 for w in words if w in SECURITY_WORDS_SET)

            tld_sub_path = 0.0
            if tld and tld in SUSPICIOUS_TLDS_SET:
                tld_sub_path = 1.0
            elif tld and len(host_parts) > 2 and tld in ".".join(host_parts[:-2]):
                tld_sub_path = 1.0

            row = [
                float(len(u)),
                float(len(clean_host)),
                float(len(path)),
                float(len(query)),
                1.0 if scheme == "https" else 0.0,
                is_ip,
                float(u.count(".")),
                float(max(0, len(host_parts) - 2)),
                float(u.count("-")),
                float(clean_host.count("-")),
                float(u.count("@")),
                float(u.count("?")),
                float(u.count("&")),
                float(u.count("=")),
                float(u.count("/")),
                float(u.count("%")),
                1.0 if "//" in full_u.replace("://", "") else 0.0,
                1.0 if tld in SUSPICIOUS_TLDS_SET else 0.0,
                1.0 if clean_host in SHORTENERS_SET else 0.0,
                float(sum(c.isdigit() for c in u) / max(1, len(u))),
                float(sum(c.isdigit() for c in clean_host) / max(1, len(clean_host))),
                shannon_entropy(clean_host),
                shannon_entropy(u),
                float(phish_hits),
                brand_flag,
                1.0 if "http" in path.lower() else 0.0,
                1.0 if "xn--" in clean_host else 0.0,
                float(len(words)),
                float(max((len(w) for w in words), default=0)),
                float(sum(len(w) for w in words) / max(1, len(words))),
                1.0 if port is not None else 0.0,
                tld_sub_path
            ]
            features.append(row)
        return np.array(features, dtype=np.float32)

