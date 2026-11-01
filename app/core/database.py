
import os

from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# MongoDB Atlas connection string
MONGODB_URI = os.getenv("MONGODB_URI")

# Database variables
client = None
db = None
users_collection = None
history_collection = None


# Connect to MongoDB
if MONGODB_URI:
    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
        )

        # Separate database for TrustGuard AI V2
        db = client["trustguard_v2"]

        # Collections
        users_collection = db["users"]
        history_collection = db["history"]

        print("MongoDB client initialized for TrustGuard AI V2.")

    except Exception as e:
        print(f"MongoDB client initialization warning: {e}")

else:
    print(
        "Warning: MONGODB_URI is not configured. "
        "MongoDB features will be unavailable."
    )


# Test MongoDB connection
def test_database_connection():
    if client is None:
        return False

    try:
        client.admin.command("ping")
        print("MongoDB connection successful.")
        return True

    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False
