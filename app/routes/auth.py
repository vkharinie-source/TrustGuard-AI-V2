
import hashlib
import secrets

from fastapi import APIRouter, HTTPException, status
from pydantic import AliasChoices, BaseModel, EmailStr, Field
from pymongo.errors import DuplicateKeyError

from app.core.database import users_collection
from app.core.security import create_access_token


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()

    return password_hash, salt


def verify_password(
    password: str,
    stored_hash: str,
    salt: str,
) -> bool:
    calculated_hash, _ = hash_password(password, salt)

    return secrets.compare_digest(
        calculated_hash,
        stored_hash,
    )


# ============================================================
# REQUEST MODELS
# ============================================================

class RegisterRequest(BaseModel):
    full_name: str = Field(
        default="",
        validation_alias=AliasChoices(
            "fullName",
            "full_name",
            "name",
        ),
    )

    email: EmailStr
    password: str

    confirm_password: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "confirmPassword",
            "confirm_password",
        ),
    )


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ============================================================
# DATABASE VALIDATION
# ============================================================

def get_users_collection():
    if users_collection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User database is unavailable. Please try again later.",
        )

    return users_collection


# ============================================================
# REGISTER
# ============================================================

@router.post("/register")
async def register(data: RegisterRequest):

    full_name = data.full_name.strip()
    email = str(data.email).lower().strip()
    password = data.password

    if not full_name:
        raise HTTPException(
            status_code=400,
            detail="Full name is required.",
        )

    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters.",
        )

    if data.confirm_password is not None:
        if password != data.confirm_password:
            raise HTTPException(
                status_code=400,
                detail="Passwords do not match.",
            )

    collection = get_users_collection()

    try:
        existing_user = collection.find_one(
            {"email": email}
        )

        if existing_user:
            raise HTTPException(
                status_code=409,
                detail="An account with this email already exists.",
            )

        password_hash, salt = hash_password(password)

        user_document = {
            "full_name": full_name,
            "email": email,
            "password_hash": password_hash,
            "salt": salt,
        }

        collection.insert_one(user_document)

        token = create_access_token(email)

        return {
            "success": True,
            "message": "Account created successfully.",
            "token": token,
            "user": {
                "full_name": full_name,
                "email": email,
            },
        }

    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"MongoDB register error: {e}")

        raise HTTPException(
            status_code=503,
            detail="Unable to create your account right now.",
        )


# ============================================================
# LOGIN
# ============================================================

@router.post("/login")
async def login(data: LoginRequest):

    email = str(data.email).lower().strip()
    password = data.password

    collection = get_users_collection()

    try:
        user = collection.find_one(
            {"email": email}
        )

    except Exception as e:
        print(f"MongoDB login error: {e}")

        raise HTTPException(
            status_code=503,
            detail="Unable to access your account right now.",
        )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    if not verify_password(
        password,
        user["password_hash"],
        user["salt"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    token = create_access_token(email)

    return {
        "success": True,
        "message": "Login successful.",
        "token": token,
        "user": {
            "full_name": user.get("full_name", "User"),
            "email": user["email"],
        },
    }


# ============================================================
# AUTH TEST
# ============================================================

@router.get("/test")
async def auth_test():

    return {
        "success": True,
        "message": "Authentication API is working.",
    }
