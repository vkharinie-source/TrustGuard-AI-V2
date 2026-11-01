import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# MongoDB Atlas connection string with cloud fallback
MONGODB_URI = os.getenv("MONGODB_URI") or (
    "mongodb+srv://trustguard_v2_user:a5edaceab8f9dfe36d9fc401858108e6"
    "@cluster0.bfvh76q.mongodb.net/trustguard_v2?retryWrites=true&w=majority"
)

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
            serverSelectionTimeoutMS=4000,
            connectTimeoutMS=4000,
        )
        # Separate database for TrustGuard AI V2
        db = client["trustguard_v2"]

        # Collections
        users_collection = db["users"]
        history_collection = db["history"]
        print("MongoDB client initialized for TrustGuard AI V2.")
    except Exception as e:
        print(f"MongoDB client initialization warning: {e}")
        client = None
        db = None
        users_collection = None
        history_collection = None
else:
    print("Warning: MONGODB_URI is not configured. Running in standalone memory mode.")


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
