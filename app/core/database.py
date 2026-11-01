import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

client = None
db = None
users_collection = None
history_collection = None

if MONGODB_URI:
    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=3000,
        )
        db = client["trustguard"]
        users_collection = db["users"]
        history_collection = db["history"]
    except Exception as e:
        print(f"MongoDB client initialization warning: {e}")
else:
    print("Notice: MONGODB_URI not configured. Running in local standalone mode.")


def test_database_connection():
    if not client:
        return False
    try:
        client.admin.command("ping")
        return True
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False