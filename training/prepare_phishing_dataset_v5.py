"""
TrustGuard AI - Phishing URL Dataset Preparation V5
Module: Website / Phishing URL Detection

Audits and augments Training.parquet with:
- Modern legitimate domains (AI platforms, developer hubs, news, academic, government)
- Legitimate URLs with complex paths and query parameters
- Contemporary phishing archetypes (cloud-hosting abuse, brand typosquatting, misleading subdomains, IP phishing)
- Preserves Testing.parquet strictly untouched as the independent test evaluation set.
"""

import json
from pathlib import Path
import pandas as pd
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets" / "phishing_urls"
TRAIN_PARQUET = DATA_DIR / "Training.parquet"
TEST_PARQUET = DATA_DIR / "Testing.parquet"
OUTPUT_PARQUET = DATA_DIR / "training_v5.parquet"
REPORT_DIR = BASE_DIR / "reports"

# 1. Additional Legitimate URLs (Modern, diverse, complex structures, unfamiliar TLDs, no reliance on HTTPS alone)
ADDITIONAL_LEGITIMATE = [
    # Modern AI & ML
    "https://chatgpt.com",
    "https://chatgpt.com/c/6701a2b3-c4d5-8001-9f23-112233445566",
    "https://openai.com/index/hello-gpt-4o/",
    "https://openai.com/api/pricing/",
    "https://claude.ai",
    "https://claude.ai/chat/new",
    "https://anthropic.com/research",
    "https://deepseek.com",
    "https://chat.deepseek.com",
    "https://huggingface.co/models?pipeline_tag=text-classification",
    "https://huggingface.co/datasets/sid321axn/malicious-urls-dataset",
    "https://platform.openai.com/docs/guides/text-generation",
    "https://console.anthropic.com/settings/billing",

    # Tech, Developer & Cloud documentation
    "https://github.com/fastapi/fastapi/releases/tag/0.110.0",
    "https://raw.githubusercontent.com/psf/requests/main/README.md",
    "https://gitlab.com/gitlab-org/gitlab/-/issues/432100",
    "https://stackoverflow.com/questions/509211/understanding-slice-notation",
    "https://serverfault.com/questions/100234/how-to-configure-nginx-reverse-proxy",
    "https://superuser.com/questions/342111/how-to-fix-dns-lookup-failure",
    "https://docs.python.org/3.12/library/urllib.parse.html#urlparse",
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_mfa.html",
    "https://cloud.google.com/storage/docs/json_api/v1/objects/get",
    "https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy",
    "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
    "https://hub.docker.com/_/python/tags?page=1&name=3.12",
    "https://pypi.org/project/scikit-learn/#history",
    "https://npmjs.com/package/react-router-dom/v/6.22.3",
    "https://crates.io/crates/tokio",
    "https://golang.org/doc/effective_go",
    "https://rust-lang.org/learn",
    "https://archlinux.org/packages/core/x86_64/linux/",
    "https://kernel.org/pub/linux/kernel/v6.x/ChangeLog-6.8",

    # Media, Knowledge & Culture
    "https://en.wikipedia.org/wiki/Phishing",
    "https://en.wikipedia.org/wiki/Transport_Layer_Security",
    "https://en.wikipedia.org/wiki/Mutual_authentication",
    "https://wikidata.org/wiki/Q131393",
    "https://archive.org/details/nasa_blue_marble",
    "https://gutenberg.org/ebooks/1342",
    "https://imdb.com/title/tt0133093/fullcredits",
    "https://rottentomatoes.com/m/matrix/reviews",
    "https://theverge.com/2024/1/15/tech-cybersecurity-news",
    "https://wired.com/story/security-threat-intelligence-report/",
    "https://arstechnica.com/information-technology/2024/02/kernel-flaw-patch/",
    "https://techcrunch.com/category/startups/",
    "https://reuters.com/technology/artificial-intelligence-regulations/",
    "https://apnews.com/hub/technology",
    "https://bbc.com/news/technology-68123456",
    "https://nature.com/articles/d41586-024-00123-x",
    "https://scientificamerican.com/article/the-science-of-cryptography/",

    # Education & Government
    "https://mit.edu/education/",
    "https://csail.mit.edu/research/cybersecurity-at-mit",
    "https://stanford.edu/academics/undergraduate-study/",
    "https://ox.ac.uk/admissions/graduate/courses",
    "https://cam.ac.uk/research/news/machine-learning-in-healthcare",
    "https://csrc.nist.gov/publications/detail/sp/800-63/3/final",
    "https://cisa.gov/known-exploited-vulnerabilities-catalog",
    "https://irs.gov/payments/your-online-account",
    "https://nasa.gov/mission/james-webb-space-telescope/",
    "https://who.int/emergencies/disease-outbreak-news",

    # Commerce, Finance & Services (Official)
    "https://amazon.com/gp/help/customer/display.html?nodeId=508510",
    "https://support.apple.com/en-us/HT204144",
    "https://support.google.com/accounts/answer/185839",
    "https://myaccount.google.com/security",
    "https://help.netflix.com/en/node/412",
    "https://support.microsoft.com/en-us/account-billing",
    "https://paypal.com/us/smarthelp/contact-us",
    "https://stripe.com/docs/security/guide",
    "https://chase.com/personal/security-and-fraud",
    "https://wellsfargo.com/privacy-security/fraud/",
    "https://shopify.com/pricing",
    "https://cloudflare.com/learning/ssl/what-is-an-ssl-certificate/",
    "https://reddit.com/r/cybersecurity/comments/192abc/best_practices_for_url_verification/",
    "https://linkedin.com/learning/it-security-foundations",

    # Unusual-looking but legitimate URLs (complex query parameters, unusual paths, subdomains)
    "http://info.cern.ch/hypertext/WWW/TheProject.html",
    "http://neverssl.com/online_status.html",
    "https://maps.google.com/maps?q=37.7749,-122.4194&z=12",
    "https://translate.google.com/?sl=auto&tl=es&text=cybersecurity&op=translate",
    "https://youtube.com/watch?v=kJQP7kiw5Fk&t=120s&list=PL123456",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7095448/?report=classic",
    "https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=adversarial+machine+learning&btnG=",
    "https://en.m.wikipedia.org/w/index.php?title=URL&action=history&offset=20240101",
    "https://web.archive.org/web/20240101000000*/https://www.google.com",
    "https://bugzilla.mozilla.org/show_bug.cgi?id=1874221",
    "https://gitlab.com/search?search=phishing_detector&nav_source=navbar&project_id=123",
]

# 2. Contemporary Phishing URLs (Diverse patterns: brand impersonation, cloud hosting, subdomains, token queries)
ADDITIONAL_PHISHING = [
    # Brand Impersonation on Fake Domains / Typosquats
    "http://paypal.login-verify-account-security.xyz/signin",
    "https://paypal.verification-center-alert992.com/secure/login.php",
    "http://paypal-resolution-center-case4492.com/auth/login",
    "https://paypal-identity-confirm-user.online/webapps/mpp/home",
    "http://apple-id-verify.support-device.top/login.html",
    "https://appleid.apple.com-manage-account.info/auth/verify",
    "http://apple-icloud-find-lost-device-auth.xyz/view",
    "https://chase-online-banking-security-update.com/auth/signin",
    "http://chase-identity-reconfirmation.net/verify_credentials.php",
    "https://wellsfargo-verify-identity.com/auth/login.jsp",
    "http://wellsfargo-security-alert-case091.com/update-info",
    "https://bankofamerica-auth-confirm.top/online/login.html",
    "http://citibank-secure-card-update.xyz/citi/auth",
    "https://netflix-account-suspended-renew.com/your-account",
    "http://netflix-billing-update-center.online/signin",
    "https://portal-netflix-update.com/login.php",
    "http://amazon-account-locked-verification.xyz/ap/signin",
    "https://amazon-prime-payment-failed-alert.info/order/update",
    "http://microsoft-office365-password-reset.top/auth/login",
    "https://microsoft-onedrive-document-share.online/login.aspx",
    "http://login.live.com.user-auth-verify.xyz/oauth/v2",
    "https://secure-microsoft-online.oa.r.appspot.com/login",
    "http://google-drive-shared-invoice-view.top/docs/download",
    "https://accounts.google.com.user-identity-check.info/signin",

    # Cloud Platform Abuse (Appspot, Cloudflare Pages, Firebase, Storage, AWS S3)
    "https://storage.cloud.google.com/malicious-phish-bucket/secure-banking-login.html",
    "https://storage.cloud.google.com/phish-assets-992/apple-id-login/index.html",
    "https://auth-portal-verify.firebaseapp.com/login.html",
    "https://chase-auth-service.firebaseapp.com/sign-in",
    "https://paypal-account-center.web.app/verification",
    "https://secure-login-portal.pages.dev/auth/index.html",
    "https://wellsfargo-alert.pages.dev/login",
    "https://s3.amazonaws.com/phishing-payload-drop/invoice-login.html",
    "https://s3.us-west-2.amazonaws.com/secure-document-portal/signin.htm",
    "https://sharedpont77wwjxjx.df.r.appspot.com/office/login.php",
    "https://onedrive-secure-file-share.workers.dev/auth",

    # Raw IP & Port Obfuscation
    "http://192.168.1.100:8080/secure-banking-login/index.html",
    "http://45.33.32.156:8000/paypal/login.php",
    "http://185.220.101.5/bank-auth/account-verification.html",
    "http://103.251.167.22/chase/signon.html",
    "http://91.240.118.172:443/wells-fargo/signin.php",
    "http://194.87.139.44/apple-id/login",

    # Deceptive Subdomain Hierarchy & Typosquatting
    "https://bancodevenezuela.com.fakeauth.net/portal/login",
    "http://paypal.com.account-update.cgi-bin.webscr.attacker-domain.xyz",
    "http://secure.chase.com.accounts-verification-online.com/auth",
    "http://signin.ebay.com.ws-ebayisapi.security-update-99.org",
    "https://login.yahoo.com.auth-service-redirect.info/login.php",
    "http://myetherwallet-phishing-scam-site.ru/unlock",
    "https://steamcommunity-trade-offer-login.com/tradeoffer/new",
    "http://steamcommunity.com.partner-trade-confirm.xyz/trade",
    "https://binance-security-kyc-verify.top/user/login",
    "http://coinbase-wallet-restore-passphrase.xyz/recover",
    "https://metamask-seed-phrase-verification.info/import",
    "http://irs-tax-refund-claim2024.com/taxpayer/refund.php",
    "http://irs-direct-deposit-stimulus.xyz/claim/identity",
    "http://covid19-relief-fund-application.online/apply/step1",

    # Token/Credential Query Harvesting & Open Redirect Tricks
    "http://secure-banking-alert.com/login.php?user=victim@company.com&token=48201a&action=verify",
    "https://verify-billing-info.org/checkout.php?invoice_id=99281&email=admin@target.org",
    "http://account-reconfirm.net/portal.php?auth_token=8923bc&dest=online_banking",
    "http://redirect-service-update.xyz/go.php?url=http://fake-login-bank.com",
    "https://secure-login-chase-update-account.com",
    "http://portal-netflix-update.com/login.php",
]


def prepare_dataset():
    print("=" * 60)
    print("PREPARING ENHANCED PHISHING DATASET V5")
    print("=" * 60)

    if not TRAIN_PARQUET.exists() or not TEST_PARQUET.exists():
        raise FileNotFoundError("Missing baseline parquet datasets.")

    train_base = pd.read_parquet(TRAIN_PARQUET)
    test_base = pd.read_parquet(TEST_PARQUET)

    print(f"Base Training.parquet rows: {len(train_base)}")
    print(f"Base Testing.parquet rows:  {len(test_base)}")

    # Keep only URL and status from base parquet
    train_clean = train_base[["url", "status"]].copy()
    train_clean["url"] = train_clean["url"].astype(str).str.strip()
    train_clean["status"] = train_clean["status"].astype(str).str.lower().str.strip()

    # Deduplicate within train
    initial_count = len(train_clean)
    train_clean = train_clean.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"Deduplicated base training: {initial_count} -> {len(train_clean)}")

    # Build augmentation dataframe
    aug_rows = []
    for u in ADDITIONAL_LEGITIMATE:
        aug_rows.append({"url": u.strip(), "status": "legitimate"})
    for u in ADDITIONAL_PHISHING:
        aug_rows.append({"url": u.strip(), "status": "phishing"})

    aug_df = pd.DataFrame(aug_rows)

    # Check that none of the augmented URLs exist in Testing.parquet
    test_urls = set(test_base["url"].astype(str).str.strip())
    aug_df = aug_df[~aug_df["url"].isin(test_urls)].reset_index(drop=True)

    # Combine base training with augmentation
    combined = pd.concat([train_clean, aug_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)

    print(f"Total V5 Training Samples: {len(combined)}")
    print(f"Status distribution:\n{combined['status'].value_counts().to_dict()}")

    # Ensure Testing.parquet remains untouched
    assert len(test_base) == 3772, "Testing.parquet row count must remain exactly 3,772"

    # Save training_v5.parquet
    combined.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved enhanced dataset to: {OUTPUT_PARQUET}")

    # Save dataset audit report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {
        "dataset_name": "Phishing URLs Dataset V5",
        "output_path": str(OUTPUT_PARQUET),
        "base_training_count": len(train_clean),
        "additional_legitimate_count": len(ADDITIONAL_LEGITIMATE),
        "additional_phishing_count": len(ADDITIONAL_PHISHING),
        "total_augmented_training_samples": len(combined),
        "class_distribution": combined["status"].value_counts().to_dict(),
        "preserved_heldout_test_count": len(test_base),
        "heldout_test_distribution": test_base["status"].value_counts().to_dict(),
        "overlap_with_heldout_test_urls": len(set(combined["url"]).intersection(test_urls))
    }

    with open(REPORT_DIR / "website_detection_dataset_v5_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"Audit report saved to: {REPORT_DIR / 'website_detection_dataset_v5_audit.json'}")
    return audit


if __name__ == "__main__":
    prepare_dataset()
