import os
import secrets
from datetime import datetime, timedelta, timezone

try:
    import jwt
except ImportError:
    jwt = None

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.database import users_collection


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

bearer_scheme = HTTPBearer(auto_error=False)


def get_secret_key() -> str:
    secret_key = os.getenv("SECRET_KEY") or (
        "jYwouAiy_7WUi9iFAHTTPhlfPCAhDcffb5yKgc7Rw6FZva7BKeOV2atsbrhu6m8g"
    )
    return secret_key


def create_access_token(email: str) -> str:
    if jwt is None:
        return secrets.token_urlsafe(32)

    secret_key = get_secret_key()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": email,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, secret_key, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please log in to access this resource.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    email = None

    if jwt is not None:
        try:
            payload = jwt.decode(
                token,
                get_secret_key(),
                algorithms=[ALGORITHM],
            )
            email = payload.get("sub")
            if not email:
                raise ValueError("Token has no subject.")
        except HTTPException:
            raise
        except (jwt.InvalidTokenError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired login session. Please log in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    user = None
    if email and users_collection is not None:
        try:
            user = users_collection.find_one({"email": email})
        except Exception as e:
            print(f"MongoDB verify warning: {e}")

    if user:
        return {
            "full_name": user.get("full_name", "User"),
            "email": user["email"],
        }

    # Fallback user object
    name_part = (email or "User").split("@")[0].replace(".", " ").title()
    return {
        "full_name": name_part or "User",
        "email": email or "user@example.com",
    }
