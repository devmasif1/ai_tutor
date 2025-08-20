from fastapi import FastAPI, HTTPException, Depends, status 
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from pymongo import MongoClient
import bcrypt
from datetime import datetime, timedelta
import jwt
import os
import uuid
from typing import  Dict, Any
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# FastAPI app
app = FastAPI(title="AI Tutor Backend - Quiz-Based Tracking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
SECRET_KEY = os.getenv("MY_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("MY_SECRET_KEY is not set. Check your .env")

ALGORITHM = os.getenv("ALGORITHM")
TOKEN_EXPIRE_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS"))

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client.ai_tutor_db

# Collections
users = db.users
quiz_reports = db.quiz_reports  # Only quiz-based tracking now

# Pydantic models
class UserRegister(BaseModel):
    username: str
    password: str
    email: str

class UserLogin(BaseModel):
    username: str
    password: str

class QuizReport(BaseModel):
    topic: str
    subject: str
    total_marks: int
    scored_marks: int
    feedback: str
    quiz_data: Dict[str, Any]  # Store quiz questions/answers

class User(BaseModel):
    id: str
    username: str
    email: str

# Helper functions
def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
  
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def get_user_vector_collection(user_id: str):
    """Get the vector collection for a specific user"""
    collection_name = f"user_vectors_{user_id}"
    return db[collection_name]

# Routes
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/register")
async def register(user_data: UserRegister):
    if users.find_one({"$or": [{"username": user_data.username}, {"email": user_data.email}]}):
        raise HTTPException(status_code=400, detail="User already exists")
    
    user_id = str(uuid.uuid4())
    user_doc = {
        "_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": hash_password(user_data.password),  
        "created_at": datetime.utcnow(),
        "vector_collection": f"user_vectors_{user_id}"
    }
    
    users.insert_one(user_doc)
    token = create_token(user_id)
    
    # Initialize empty vector collection for user
    try:
        vector_collection = get_user_vector_collection(user_id)
        vector_collection.create_index([
            ("user_id", 1),
            ("doc_type", 1),
            ("created_at", -1)
        ])
        print(f"[INFO] Created vector collection for user: {user_id}")
    except Exception as e:
        print(f"[WARNING] Could not initialize vector collection for user {user_id}: {e}")
    
    return {"message": "User registered", "token": token, "user": {"id": user_id, "username": user_data.username}}

@app.post("/login")
async def login(user_data: UserLogin):
    user = users.find_one({"username": user_data.username})
    if not user or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(user["_id"])
    return {"message": "Login successful", "token": token, "user": {"id": user["_id"], "username": user["username"]}}

@app.get("/me")
async def get_current_user(user_id: str = Depends(verify_token)):
    user = users.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"id": user["_id"], "username": user["username"], "email": user["email"]}

# NEW Quiz-based tracking endpoints
@app.post("/save_quiz_report")
async def save_quiz_report(report_data: QuizReport, user_id: str = Depends(verify_token)):
    """Save quiz report to database"""
    quiz_report_doc = {
        "_id": str(uuid.uuid4()),
        "user_id": user_id,
        "topic": report_data.topic,
        "subject": report_data.subject,
        "total_marks": report_data.total_marks,
        "scored_marks": report_data.scored_marks,
        "feedback": report_data.feedback,
        "quiz_data": report_data.quiz_data,
        "timestamp": datetime.utcnow()
    }
    
    quiz_reports.insert_one(quiz_report_doc)
    return {"message": "Quiz report saved successfully"}

@app.get("/get_quiz_reports/{subject}")
async def get_quiz_reports_by_subject(subject: str, user_id: str = Depends(verify_token)):
    """Get all quiz reports for a specific subject"""
    reports = list(quiz_reports.find(
        {"user_id": user_id, "subject": subject}
    ).sort("timestamp", -1))
    
    return {"subject": subject, "reports": reports, "total_quizzes": len(reports)}

@app.get("/get_all_subjects")
async def get_all_subjects(user_id: str = Depends(verify_token)):
    """Get all unique subjects for the user"""
    subjects = quiz_reports.distinct("subject", {"user_id": user_id})
    return {"subjects": subjects}

@app.get("/generate_subject_report/{subject}")
async def generate_subject_report(subject: str, user_id: str = Depends(verify_token)):
    """Generate comprehensive report for a specific subject"""
    reports = list(quiz_reports.find(
        {"user_id": user_id, "subject": subject}
    ).sort("timestamp", -1))
    
    if not reports:
        raise HTTPException(status_code=404, detail=f"No quiz reports found for subject: {subject}")
    
    # Calculate statistics
    total_quizzes = len(reports)
    total_marks_possible = sum(report["total_marks"] for report in reports)
    total_marks_scored = sum(report["scored_marks"] for report in reports)
    average_percentage = (total_marks_scored / total_marks_possible) * 100 if total_marks_possible > 0 else 0
    
    # Get all topics covered
    topics_covered = list(set(report["topic"] for report in reports))
    
    # Get recent performance (last 5 quizzes)
    recent_performance = []
    for report in reports[:5]:
        percentage = (report["scored_marks"] / report["total_marks"]) * 100 if report["total_marks"] > 0 else 0
        recent_performance.append({
            "topic": report["topic"],
            "percentage": round(percentage, 2),
            "date": report["timestamp"]
        })
    
    return {
        "subject": subject,
        "total_quizzes": total_quizzes,
        "average_percentage": round(average_percentage, 2),
        "total_marks_possible": total_marks_possible,
        "total_marks_scored": total_marks_scored,
        "topics_covered": topics_covered,
        "recent_performance": recent_performance,
        "all_reports": reports
    }

@app.delete("/clear_quiz_reports")
async def clear_quiz_reports(user_id: str = Depends(verify_token)):
    """Clear all quiz reports for the user"""
    result = quiz_reports.delete_many({"user_id": user_id})
    return {"message": f"Cleared {result.deleted_count} quiz reports"}





if __name__ == "__main__":
    import uvicorn

    host = "0.0.0.0"
    port = int(os.getenv("PORT", 8000))  # Railway provides $PORT

    print("🚀 Starting AI Tutor Backend with Quiz-Based Tracking")
    print(f"📊 MongoDB URI: {MONGO_URI}")

    uvicorn.run(app, host=host, port=port)
