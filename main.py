from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os
import uuid
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
import shutil
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()
security = HTTPBasic()

# Configuration from .env
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin_password")
# All storage paths are local to the 'backend' folder by default
STORAGE_DIR = os.getenv("STORAGE_DIR", "outputs")
VOICES_DIR = os.getenv("VOICES_DIR", "voices")
HISTORY_FILE = os.getenv("HISTORY_FILE", "history.json")

# Ensure local directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication logic
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != API_USERNAME or credentials.password != API_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Load the Turbo model (global)
model = None

def get_model():
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ChatterboxTurboTTS on {device}...")
        model = ChatterboxTurboTTS.from_pretrained(device=device)
    return model

def save_to_history(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
    
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

@app.post("/generate-tts")
async def generate_tts(
    text: str = Form(...),
    audio_prompt: UploadFile = File(None),
    voice_id: str = Form(None), # Option to use a previously saved voice
    username: str = Depends(authenticate)
):
    m = get_model()
    
    prompt_path = None
    should_cleanup_prompt = False
    
    if audio_prompt:
        # Temporary prompt for this request
        temp_id = str(uuid.uuid4())
        prompt_path = os.path.join(STORAGE_DIR, f"temp_prompt_{temp_id}.wav")
        with open(prompt_path, "wb") as buffer:
            shutil.copyfileobj(audio_prompt.file, buffer)
        should_cleanup_prompt = True
    elif voice_id:
        # Use a stored voice clone from the local 'voices' directory
        voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
        if os.path.exists(voice_path):
            prompt_path = voice_path
        else:
            raise HTTPException(status_code=404, detail=f"Voice clone '{voice_id}' not found in local storage.")
    
    try:
        if prompt_path:
            # Generate speech using the reference voice (cloning)
            wav = m.generate(text, audio_prompt_path=prompt_path)
        else:
            return {"error": "audio_prompt or voice_id is required for Chatterbox-Turbo"}

        out_id = str(uuid.uuid4())
        out_filename = f"gen_{out_id}.wav"
        out_path = os.path.join(STORAGE_DIR, out_filename)
        
        ta.save(out_path, wav, m.sr)
        
        # Save to history
        history_entry = {
            "id": out_id,
            "text": text,
            "filename": out_filename,
            "voice_used": voice_id if voice_id else "uploaded_sample",
            "timestamp": datetime.now().isoformat(),
            "user": username
        }
        save_to_history(history_entry)
        
        return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")
    
    finally:
        if should_cleanup_prompt and prompt_path and os.path.exists(prompt_path):
            os.remove(prompt_path)

@app.post("/voices/upload")
async def upload_voice_clone(
    name: str = Form(...),
    file: UploadFile = File(...),
    username: str = Depends(authenticate)
):
    """Saves a voice sample locally to be used for cloning later."""
    # Sanitize name to avoid path traversal
    safe_name = "".join([c for c in name if c.isalnum() or c in (" ", "-", "_")]).strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")
    
    file_path = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"status": "Voice clone saved locally", "voice_id": safe_name}

@app.get("/voices")
async def list_voices(username: str = Depends(authenticate)):
    """Lists all locally stored voice clones."""
    if not os.path.exists(VOICES_DIR):
        return []
    voices = [f.replace(".wav", "") for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
    return {"voices": voices}

@app.get("/history")
async def get_history(username: str = Depends(authenticate)):
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    return history

@app.get("/download/{filename}")
async def download_file(filename: str, username: str = Depends(authenticate)):
    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.get("/")
def read_root():
    return {"status": "Voice Backend Running", "model": "ResembleAI/chatterbox-turbo", "auth": "enabled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
