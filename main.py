import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Response, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from openai import OpenAI

# Core project imports
from .database import SessionLocal, User, AnalysisReport
from .shot_analyzer import process_video

app = FastAPI()
client = OpenAI(api_key="sk-proj-YOUR_KEY") 

# Absolute paths for Linux system
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

for folder in [OUTPUTS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- AUTHENTICATION ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user_id = request.cookies.get("session_user")
    if user_id and user_id != "None":
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register")
async def register(data: dict, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data['username']).first():
        raise HTTPException(status_code=400, detail="Username taken")
    # Restored email handling
    new_user = User(username=data['username'], email=data.get('email'), password_hash=data['password'])
    db.add(new_user)
    db.commit()
    return {"msg": "Registration successful"}

@app.post("/login")
async def login(data: dict, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data['username'], User.password_hash == data['password']).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    response.set_cookie(key="session_user", value=str(user.id), httponly=True, samesite="lax")
    return {"msg": "Login successful"}

@app.get("/logout")
async def logout(response: Response):
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("session_user")
    return resp

# --- CORE APP ---
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("session_user")
    if not user_id or user_id == "None":
        return RedirectResponse(url="/", status_code=303)
    
    reports = db.query(AnalysisReport).filter(AnalysisReport.user_id == int(user_id)).all()
    return templates.TemplateResponse("index.html", {"request": request, "reports": reports})

@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    user_id = request.cookies.get("session_user")
    if not user_id: raise HTTPException(status_code=401, detail="Unauthorized")

    fid = str(uuid.uuid4())
    in_p = str(BASE_DIR / "uploads" / f"{fid}_{file.filename}")
    out_p = str(OUTPUTS_DIR / f"out_{fid}.mp4")
    rep_p = str(REPORTS_DIR / f"rep_{fid}.pdf")
    
    os.makedirs(BASE_DIR / "uploads", exist_ok=True)
    with open(in_p, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Run MediaPipe and Custom UCF101 Model
    angle, feedback = process_video(in_p, out_p, rep_p)
    
    # Run OpenAI Generative Coaching
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a cricket expert. Provide a technical tip based on metrics."},
                {"role": "user", "content": f"Angle: {angle}, Feedback: {feedback}"}
            ]
        )
        feedback.append(res.choices[0].message.content)
    except:
        # Fallback message removed as requested
        pass 

    new_report = AnalysisReport(
        user_id=int(user_id), 
        video_path=f"/outputs/out_{fid}.mp4", 
        report_path=f"/reports/rep_{fid}.pdf", 
        shot_type=feedback[0]
    )
    db.add(new_report)
    db.commit()

    return {"video_url": new_report.video_path, "report_url": new_report.report_path, "feedback": feedback}