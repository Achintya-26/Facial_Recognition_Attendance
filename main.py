import os
import cv2
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Direct import from train.py for consistency
from train import recognize_faces, load_embeddings_database, generate_embeddings_database
logger.info("Using face_recognition module for face detection")

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Attendance API",
    description="API for facial recognition-based attendance tracking",
    version="1.0.0"
)

# Setup CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local frontend URL
        "https://csi-portal.vercel.app/",  # Production frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
ATTENDANCE_IMAGES_DIR = BASE_DIR / "attendance_images"
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
ATTENDANCE_IMAGES_DIR.mkdir(exist_ok=True)

class SupabaseClient:
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        logger.info("Initialized Supabase client")
    
    def insert_attendance(self, data):
        """Insert attendance record to Supabase"""
        try:
            endpoint = f"{self.url}/rest/v1/attendance_records"
            response = requests.post(endpoint, headers=self.headers, json=data)
            
            if response.status_code in (200, 201):
                logger.info(f"Successfully recorded attendance for {data['name']}")
                return {"success": True, "data": response.json()}
            else:
                logger.error(f"Supabase error: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Failed to insert attendance: {e}")
            return {"success": False, "error": str(e)}
    
    def get_user_by_name(self, full_name):
        """Get user profile by full name"""
        try:
            endpoint = f"{self.url}/rest/v1/user_profiles"
            params = {"full_name": f"eq.{full_name}", "select": "id,full_name,role"}
            
            response = requests.get(endpoint, headers=self.headers, params=params)
            
            if response.status_code == 200:
                users = response.json()
                if users and len(users) > 0:
                    logger.info(f"Found user profile for {full_name}")
                    return {"success": True, "data": users[0]}
                else:
                    logger.warning(f"No user profile found for {full_name}")
                    return {"success": False, "error": "User not found"}
            else:
                logger.error(f"Supabase error: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return {"success": False, "error": str(e)}

# Initialize Supabase client with environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("⚠️ SUPABASE_URL or SUPABASE_KEY not set in .env file. Database updates will be simulated.")

supabase_client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

@app.on_event("startup")
async def startup_event():
    """Load face embeddings database when the app starts"""
    try:
        load_embeddings_database()
        logger.info("Successfully loaded face embeddings database")
    except Exception as e:
        logger.error(f"Error loading face database: {e}")
        logger.info("The app will try to load the database when needed")

@app.get("/")
async def root():
    """API health check endpoint"""
    return {"status": "online", "message": "Face Recognition Attendance API"}

@app.post("/update-database")
async def update_database():
    """Endpoint to refresh the face encodings database"""
    try:
        database = generate_embeddings_database()
        return {
            "status": "success",
            "message": f"Database updated with {len(database['names'])} faces"
        }
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        raise HTTPException(status_code=500, detail=f"Database update failed: {str(e)}")

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    domain: str = Form(...),
    event: Optional[str] = Form("Regular Attendance")
):
    """
    Recognize faces in uploaded image and record attendance
    """
    try:
        # Generate unique ID for this recognition session
        session_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        temp_file_path = TEMP_UPLOAD_DIR / f"{session_id}.jpg"
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        logger.info(f"Processing image from {temp_file_path}")
        
        # Process image for face recognition
        names, domains, confidences = recognize_faces(temp_file_path)
        
        # Save a copy of the image in the attendance images directory
        attendance_image_path = ATTENDANCE_IMAGES_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id}.jpg"
        if temp_file_path.exists():
            # Copy file to attendance images directory
            import shutil
            shutil.copy(temp_file_path, attendance_image_path)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            logger.info(f"Saved attendance image to {attendance_image_path}")
        
        # If no faces recognized
        if not names:
            return JSONResponse({
                "status": "no_match",
                "message": "No faces recognized in the image",
                "matches": []
            })
        
        # Record attendance in Supabase for each recognized face
        attendance_records = []
        current_date = datetime.now().date().isoformat()
        current_time = datetime.now().time().isoformat(timespec='seconds')
        
        for name, face_domain, confidence in zip(names, domains, confidences):
            # Only record attendance if domain matches
            if face_domain == domain:
                # Generate a unique ID for this attendance record
                record_id = str(uuid.uuid4())
                
                # Try to get the user profile by name to get the UUID
                user_id = None
                user_data = None
                
                if SUPABASE_URL and SUPABASE_KEY:
                    # Get user profile from Supabase
                    user_result = supabase_client.get_user_by_name(name)
                    if user_result.get("success", False):
                        user_data = user_result.get("data", {})
                        user_id = user_data.get("id")
                        logger.info(f"Found user ID {user_id} for {name}")
                    else:
                        logger.warning(f"Could not find user profile for {name}")
                
                # Record attendance in Supabase
                attendance_data = {
                    "id": record_id,
                    "name": name,
                    "domain": domain,
                    "date": current_date,
                    "time": current_time,
                    "confidence": float(confidence),
                    "event": event,
                    "created_at": datetime.now().isoformat(),
                    "image_reference": str(attendance_image_path.name)
                }
                
                # Add user_id if available
                if user_id:
                    attendance_data["user_id"] = user_id
                
                if SUPABASE_URL and SUPABASE_KEY:
                    # Only try to insert if we have credentials
                    result = supabase_client.insert_attendance(attendance_data)
                    recorded = result.get("success", False)
                    error = result.get("error", "") if not recorded else ""
                else:
                    # Simulate recording if no credentials
                    logger.info(f"Simulated attendance for {name} (no Supabase credentials)")
                    recorded = True
                    error = ""
                
                attendance_record = {
                    "id": record_id,
                    "name": name,
                    "confidence": float(confidence),
                    "recorded": recorded,
                    "error": error if not recorded else "",
                    "image_reference": str(attendance_image_path.name)
                }
                
                # Include user data if available
                if user_data:
                    attendance_record["user_id"] = user_id
                    attendance_record["role"] = user_data.get("role")
                
                attendance_records.append(attendance_record)
            else:
                # Face recognized but from different domain
                attendance_records.append({
                    "name": name, 
                    "domain": face_domain,
                    "confidence": float(confidence),
                    "recorded": False,
                    "error": f"Domain mismatch (expected {domain}, got {face_domain})"
                })
        
        return {
            "status": "success",
            "message": f"Recognized {len(names)} faces",
            "matches": attendance_records
        }
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# For testing - this will run if you execute the file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)