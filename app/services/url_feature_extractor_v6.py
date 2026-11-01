"""
TrustGuard AI - Phishing URL Feature Extractor V6 (Experimental)
Module: app.services.url_feature_extractor_v6

Provides URLFeatureExtractorV6 with modular feature modes for ablation study:
- 'baseline': 32 base V5 features
- 'fuzzy_brand': baseline + 4 fuzzy brand impersonation & confusable features
- 'shared_hosting': baseline + 4 shared-hosting provider awareness features
- 'short_url': baseline + 4 short-URL and entropy interaction features
- 'combined': all candidate features (46 deterministic lexical and structural features)

Purely offline, zero network requests, fast, and deterministic.
Preserves existing V5 URLFeatureExtractor completely unchanged for backward compatibility.
"""

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Curated major brands and their verified official domains (FP safeguard)
BRAND_OFFICIAL_DOMAINS = {
    "paypal": "paypal.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "amazon": "amazon.com",
    "netflix": "netflix.com",
    "chase": "chase.com",
    "wellsfargo": "wellsfargo.com",
    "citi": "citi.com",
    "bankofamerica": "bankofamerica.com",
    "binance": "binance.com",
    "coinbase": "coinbase.com",
    "steamcommunity": "steamcommunity.com",
    "roblox": "roblox.com",
    "ebay": "ebay.com",
    "skechers": "skechers.com",
    "bt": "bt.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "whatsapp": "whatsapp.com",
    "adobe": "adobe.com",
    "yahoo": "yahoo.com",
    "dropbox": "dropbox.com",
    "onedrive": "live.com",
    "outlook": "live.com",
    "icloud": "icloud.com"
}

# Verified multi-user / free / blogging / user-content hosting services
SHARED_HOSTING_PROVIDERS = [
    "tumblr.com", "blogspot.com", "blogger.com", "narod.ru", "es.tl", "totalh.net",
    "wordpress.com", "github.io", "gitlab.io", "vercel.app", "netlify.app",
    "web.app", "firebaseapp.com", "pages.dev", "workers.dev", "appspot.com",
    "site123.me", "wixsite.com", "weebly.com", "czweb.org", "000webhostapp.com",
    "s3.amazonaws.com", "storage.cloud.google.com", "cloud.google.com",
    "sites.google.com", "drive.google.com", "onedrive.live.com", "blob.core.windows.net"
]

SUSPICIOUS_TLDS_SET = {
    "xyz", "top", "click", "buzz", "work", "gq", "tk", "ml", "cf",
    "online", "site", "live", "support", "club", "icu", "cam", "zip", "mov", "ru",
    "link", "info", "monster", "rest", "bar"
}

SHORTENERS_SET = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly", "cutt.ly", "shorturl.at", "goo.gl", "cort.as", "u.gy"
}

SECURITY_WORDS_SET = {
    "login", "verify", "verification", "secure", "account", "update", "password",
    "wallet", "bank", "payment", "refund", "free", "reward", "prize", "claim",
    "bonus", "otp", "confirm", "signin", "auth", "credential", "recover", "security",
    "billing", "invoice", "validate", "support", "service", "customer", "resolution",
    "identity", "unlock", "access", "portal"
}

CONFUSABLES_MAP = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '8': 'b',
    '@': 'a', '$': 's', 'vv': 'w', 'rn': 'm'
}

GENERIC_COMPROMISED_DIRS = {
    "project", "docx", "cs", "boc", "files", "data", "doc", "invoice", "pdf"
}


def normalize_confusables(s: str) -> str:
    s = s.lower()
    for k, v in CONFUSABLES_MAP.items():
        s = s.replace(k, v)
    return s


def damerau_levenshtein(s1: str, s2: str) -> int:
    d = {}
    len1 = len(s1)
    len2 = len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,
                d[(i, j - 1)] + 1,
                d[(i - 1, j - 1)] + cost
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + cost)
    return d[(len1 - 1, len2 - 1)]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


class URLFeatureExtractorV6(BaseEstimator, TransformerMixin):
    """
    Experimental URL Feature Extractor V6 supporting modular ablation modes:
    - 'baseline': 32 original features
    - 'fuzzy_brand': +4 fuzzy brand & confusable features
    - 'shared_hosting': +4 shared-hosting provider features
    - 'short_url': +4 short-URL and entropy interaction features
    - 'combined': all 46 candidate features
    """

    def __init__(self, mode: str = "combined"):
        self.mode = mode
        self.feature_names_ = self._get_feature_names()

    def _get_feature_names(self):
        base = [
            "url_len", "host_len", "path_len", "query_len", "is_https", "is_ip",
            "dot_count", "subdomain_count", "hyphen_count", "host_hyphen_count",
            "at_count", "qm_count", "and_count", "eq_count", "slash_count",
            "percent_count", "double_slash_in_path", "suspicious_tld", "is_shortener",
            "digit_ratio_url", "digit_ratio_host", "host_entropy", "url_entropy",
            "phish_keyword_count", "brand_impersonation_flag", "http_in_path",
            "punycode", "word_count", "longest_word", "avg_word_len",
            "has_port", "tld_in_subdomain_or_path"
        ]
        if self.mode == "baseline":
            return base

        fuzzy_brand_features = [
            "fuzzy_brand_max_similarity",
            "fuzzy_brand_flag",
            "brand_prefix_suffix_flag",
            "brand_confusable_flag"
        ]

        shared_hosting_features = [
            "is_shared_hosting",
            "shared_subdomain_len",
            "shared_subdomain_entropy",
            "shared_host_brand_or_security"
        ]

        short_url_features = [
            "is_short_url",
            "path_entropy",
            "short_url_path_entropy_interaction",
            "has_random_path_token"
        ]

        compromised_domain_features = [
            "path_depth",
            "has_generic_dir"
        ]

        if self.mode == "fuzzy_brand":
            return base + fuzzy_brand_features
        elif self.mode == "shared_hosting":
            return base + shared_hosting_features
        elif self.mode == "short_url":
            return base + short_url_features
        elif self.mode == "combined":
            return base + fuzzy_brand_features + shared_hosting_features + short_url_features + compromised_domain_features
        else:
            raise ValueError(f"Unknown feature mode: {self.mode}")

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

            # Exact brand check (from V5)
            brand_flag = 0.0
            for b in BRAND_OFFICIAL_DOMAINS:
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

            # 32 Baseline features
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

            # 1. Fuzzy Brand Impersonation features
            norm_host = normalize_confusables(clean_host)
            host_tokens = re.findall(r"[a-z0-9]+", clean_host)
            norm_tokens = re.findall(r"[a-z0-9]+", norm_host)

            max_sim = 0.0
            fuz_flag = 0.0
            prefix_suffix_flag = 0.0
            confusable_flag = 0.0

            for brand, official in BRAND_OFFICIAL_DOMAINS.items():
                is_official = clean_host == official or clean_host.endswith("." + official)
                if is_official:
                    continue  # False-positive safeguard for official brand domain

                # Check confusable substitution: does normalized token equal brand while raw does not?
                for rt, nt in zip(host_tokens, norm_tokens):
                    if nt == brand and rt != brand:
                        confusable_flag = 1.0

            # Check brand prefix/suffix in host labels and tokens (e.g. skechersbuty, bt-uk, netflix-billing)
            host_labels = [p for p in clean_host.split(".") if p]
            for brand in BRAND_OFFICIAL_DOMAINS:
                if clean_host == BRAND_OFFICIAL_DOMAINS[brand] or clean_host.endswith("." + BRAND_OFFICIAL_DOMAINS[brand]):
                    continue
                for lbl in host_labels:
                    if len(lbl) > len(brand):
                        if lbl.startswith(brand + "-") or lbl.startswith(brand + "_") or lbl.startswith(brand):
                            prefix_suffix_flag = 1.0
                        elif lbl.endswith("-" + brand) or lbl.endswith("_" + brand) or lbl.endswith(brand):
                            prefix_suffix_flag = 1.0
                for t in host_tokens:
                    if len(t) > len(brand):
                        if t.startswith(brand) or t.endswith(brand):
                            length_ratio = len(brand) / len(t)
                            if length_ratio >= 0.40:
                                prefix_suffix_flag = 1.0

                # Check edit distance on tokens with comparable length
                for nt in norm_tokens:
                    len_diff = abs(len(nt) - len(brand))
                    if len_diff <= 2 and max(len(nt), len(brand)) >= 4:
                        dist = damerau_levenshtein(nt, brand)
                        sim = 1.0 - (dist / max(len(nt), len(brand)))
                        if sim > max_sim:
                            max_sim = sim
                        if dist <= 1 and nt != brand:
                            fuz_flag = 1.0

            fuzzy_vals = [
                float(max_sim),
                fuz_flag,
                prefix_suffix_flag,
                confusable_flag
            ]

            # 2. Shared Hosting Awareness features
            is_shared = 0.0
            sub_len = 0.0
            sub_ent = 0.0
            shared_brand_sec = 0.0

            for provider in SHARED_HOSTING_PROVIDERS:
                if clean_host == provider or clean_host.endswith("." + provider):
                    is_shared = 1.0
                    # Extract user-created subdomain
                    if clean_host.endswith("." + provider):
                        user_sub = clean_host[:-len(provider)-1]
                        sub_len = float(len(user_sub))
                        sub_ent = float(shannon_entropy(user_sub))
                        user_sub_tokens = re.findall(r"[a-z0-9]+", user_sub)
                        has_brand = any(b in user_sub for b in BRAND_OFFICIAL_DOMAINS)
                        has_sec = any(w in user_sub_tokens for w in SECURITY_WORDS_SET)
                        if has_brand or has_sec:
                            shared_brand_sec = 1.0
                    break

            shared_vals = [
                is_shared,
                sub_len,
                sub_ent,
                shared_brand_sec
            ]

            # 3. Short URL and Entropy features
            is_short = 1.0 if len(u) < 32 else 0.0
            p_ent = float(shannon_entropy(path))
            short_ent_inter = is_short * p_ent

            path_tokens = [t for t in re.split(r"[/._-]", path) if t]
            has_random_tok = 0.0
            for pt in path_tokens:
                if len(pt) >= 4 and shannon_entropy(pt) >= 2.2:
                    has_random_tok = 1.0
                    break

            short_vals = [
                is_short,
                p_ent,
                short_ent_inter,
                has_random_tok
            ]

            # 4. Compromised Domain features
            p_depth = float(len([seg for seg in path.split("/") if seg]))
            has_gen_dir = 1.0 if any(seg.lower() in GENERIC_COMPROMISED_DIRS for seg in path.split("/")) else 0.0

            compromised_vals = [
                p_depth,
                has_gen_dir
            ]

            if self.mode == "fuzzy_brand":
                row.extend(fuzzy_vals)
            elif self.mode == "shared_hosting":
                row.extend(shared_vals)
            elif self.mode == "short_url":
                row.extend(short_vals)
            elif self.mode == "combined":
                row.extend(fuzzy_vals)
                row.extend(shared_vals)
                row.extend(short_vals)
                row.extend(compromised_vals)

            features.append(row)

        return np.array(features, dtype=np.float32)
