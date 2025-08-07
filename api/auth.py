from fastapi import HTTPException, Header
import os
from dotenv import load_dotenv

load_dotenv()

API_KEYS = os.getenv("AUTHORIZATION")  # Store this securely in env var in production

def validate_api_key(authorization: str = Header(...)):
    try:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or token not in API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API Key")
    except Exception:
        raise HTTPException(status_code=401, detail="Authorization header is required")
