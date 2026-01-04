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
import requests
from pathlib import Path
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Authenticate with Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN and HF_TOKEN != "tu_token_huggingface_aqui":
    print(f"Logging in to Hugging Face with provided token...")
    login(token=HF_TOKEN)
else:
    print("WARNING: HF_TOKEN not set in .env. Model download might fail for gated repositories.")

app = FastAPI()
security = HTTPBasic()

# Configuration from .env
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin_password")
# All storage paths are local to the 'backend' folder by default
STORAGE_DIR = os.getenv("STORAGE_DIR", "outputs")
VOICES_DIR = os.getenv("VOICES_DIR", "voices")
VOICES_META_FILE = os.path.join(VOICES_DIR, "metadata.json")
HISTORY_FILE = os.getenv("HISTORY_FILE", "history.json")

# Ensure local directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# Initialize metadata file if it doesn't exist
if not os.path.exists(VOICES_META_FILE):
    with open(VOICES_META_FILE, "w") as f:
        json.dump({}, f)

def get_voice_metadata():
    if not os.path.exists(VOICES_META_FILE):
        return {}
    try:
        with open(VOICES_META_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_voice_metadata(metadata):
    with open(VOICES_META_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

# Default voice samples to download on first run
# Using Mozilla TTS LJSpeech public domain samples (English only)
# Users can upload their own Spanish voices using the /voices/upload endpoint
DEFAULT_VOICES = {
    "female_english": "https://github.com/mozilla/TTS/raw/master/tests/data/ljspeech/wavs/LJ001-0001.wav",
    "male_english": "https://github.com/mozilla/TTS/raw/master/tests/data/ljspeech/wavs/LJ001-0005.wav",
}

def register_default_metadata():
    """Ensures default voices have metadata entries."""
    metadata = get_voice_metadata()
    changed = False
    
    defaults_meta = {
        "female_english": {"language": "en", "gender": "female", "region": "US", "description": "Default English Female"},
        "male_english": {"language": "en", "gender": "male", "region": "US", "description": "Default English Male"}
    }
    
    for voice_id, meta in defaults_meta.items():
        if voice_id not in metadata:
            metadata[voice_id] = {
                "name": voice_id,
                "filename": f"{voice_id}.wav",
                "uploaded_by": "system",
                "uploaded_at": datetime.now().isoformat(),
                **meta
            }
            changed = True
            
    if changed:
        save_voice_metadata(metadata)

def download_default_voices():
    """Download default voice samples if they don't exist."""
    voices_path = Path(VOICES_DIR)
    
    # Register metadata for defaults
    register_default_metadata()
    
    # Check if any default voices already exist
    existing_voices = list(voices_path.glob("*.wav"))
    if existing_voices:
        print(f"Found {len(existing_voices)} existing voice(s) in {VOICES_DIR}")
        return
    
    print("Downloading default voice samples...")
    for voice_name, url in DEFAULT_VOICES.items():
        try:
            voice_file = voices_path / f"{voice_name}.wav"
            print(f"  - Downloading {voice_name}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(voice_file, "wb") as f:
                f.write(response.content)
            print(f"    ✓ Saved to {voice_file}")
        except Exception as e:
            print(f"    ✗ Failed to download {voice_name}: {e}")
    
    print("Default voices setup complete!")

def save_upload_as_wav(upload_file: UploadFile, dest_path: str):
    """
    Saves an uploaded file to the destination path as a WAV file.
    Converts MP3/M4A/etc automatically using torchaudio.
    """
    # Create a temporary file to save the original upload
    suffix = Path(upload_file.filename).suffix
    temp_path = f"temp_upload_{uuid.uuid4()}{suffix}"
    
    try:
        # Save original file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        # Load audio (torchaudio handles automatic format detection)
        # and save as WAV to destination
        waveform, sample_rate = ta.load(temp_path)
        ta.save(dest_path, waveform, sample_rate)
        
    except Exception as e:
        # Cleanup on failure
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Download default voices on startup
download_default_voices()

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
    voice_id: str = Form(None),
    language: str = Form("en"), # Default to English if not specified
    username: str = Depends(authenticate)
):
    m = get_model()
    
    prompt_path = None
    should_cleanup_prompt = False
    
    if audio_prompt:
        # Temporary prompt for this request
        temp_id = str(uuid.uuid4())
        prompt_path = os.path.join(STORAGE_DIR, f"temp_prompt_{temp_id}.wav")
        # Convert uploaded prompt to WAV
        save_upload_as_wav(audio_prompt, prompt_path)
        should_cleanup_prompt = True
    elif voice_id:
        # Use a stored voice clone from the local 'voices' directory
        voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
        if os.path.exists(voice_path):
            prompt_path = voice_path
            
            # Auto-detect language from metadata if not explicitly provided
            if language == "en": # Only override default
                meta = get_voice_metadata().get(voice_id, {})
                if meta.get("language"):
                    language = meta.get("language")
        else:
            raise HTTPException(status_code=404, detail=f"Voice clone '{voice_id}' not found in local storage.")
    
    try:
        if prompt_path:
            # Generate speech using the reference voice (cloning)
            # Pass language to fix accent issues (e.g. 'es' for Spanish)
            wav = m.generate(text, audio_prompt_path=prompt_path, language=language)
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

    except AssertionError as e:
        if "longer than 5 seconds" in str(e):
            raise HTTPException(status_code=400, detail="The audio prompt (voice sample) is too short. It must be at least 5 seconds long.")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing TTS: {str(e)}")
    
    finally:
        if should_cleanup_prompt and prompt_path and os.path.exists(prompt_path):
            os.remove(prompt_path)

@app.post("/voices/upload")
async def upload_voice_clone(
    name: str = Form(...),
    file: UploadFile = File(...),
    language: str = Form("en"),
    region: str = Form(None),
    gender: str = Form(None),
    description: str = Form(None),
    username: str = Depends(authenticate)
):
    """Saves a voice sample locally with metadata."""
    # Sanitize name to avoid path traversal
    safe_name = "".join([c for c in name if c.isalnum() or c in (" ", "-", "_")]).strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")
    
    file_path = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    try:
        save_upload_as_wav(file, file_path)
        
        # Save metadata
        metadata = get_voice_metadata()
        metadata[safe_name] = {
            "name": safe_name,
            "filename": f"{safe_name}.wav",
            "language": language,
            "region": region,
            "gender": gender,
            "description": description,
            "uploaded_by": username,
            "uploaded_at": datetime.now().isoformat()
        }
        save_voice_metadata(metadata)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file format. Could not convert to WAV. Error: {str(e)}")
    
    return {
        "status": "Voice clone saved locally", 
        "voice_id": safe_name,
        "metadata": metadata[safe_name]
    }

@app.get("/voices")
async def list_voices(username: str = Depends(authenticate)):
    """Lists all locally stored voice clones with their metadata."""
    if not os.path.exists(VOICES_DIR):
        return []
    
    voice_files = [f for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
    metadata = get_voice_metadata()
    
    result = []
    for f in voice_files:
        voice_id = f.replace(".wav", "")
        # Get metadata or default if missing
        meta = metadata.get(voice_id, {
            "name": voice_id,
            "language": "unknown",
            "region": None,
            "gender": None
        })
        # Ensure filename matches existing file
        meta["filename"] = f
        meta["id"] = voice_id
        result.append(meta)
        
    return {"voices": result}

@app.get("/history")
async def get_history(username: str = Depends(authenticate)):
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        # Sort by timestamp desc
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history
    except:
        return []

@app.delete("/history")
async def delete_history(ids: list[str] = Form(None), delete_all: bool = Form(False), username: str = Depends(authenticate)):
    """Delete history entries. Can delete specific IDs or all history."""
    if not os.path.exists(HISTORY_FILE):
        return {"status": "History empty"}
        
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
            
        if delete_all:
            # Remove all files
            for entry in history:
                fpath = os.path.join(STORAGE_DIR, entry["filename"])
                if os.path.exists(fpath):
                    os.remove(fpath)
            history = []
        elif ids:
            # Delete specific files and entries
            new_history = []
            for entry in history:
                if entry["id"] in ids:
                    fpath = os.path.join(STORAGE_DIR, entry["filename"])
                    if os.path.exists(fpath):
                        os.remove(fpath)
                else:
                    new_history.append(entry)
            history = new_history
            
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
            
        return {"status": "History updated", "remaining_count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error managing history: {str(e)}")

@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str, username: str = Depends(authenticate)):
    """Delete a generated voice clone."""
    voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    
    if not os.path.exists(voice_path):
        raise HTTPException(status_code=404, detail="Voice not found")
        
    try:
        os.remove(voice_path)
        
        # Update metadata
        metadata = get_voice_metadata()
        if voice_id in metadata:
            del metadata[voice_id]
            save_voice_metadata(metadata)
            
        return {"status": f"Voice '{voice_id}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting voice: {str(e)}")

@app.put("/voices/{voice_id}")
async def update_voice(
    voice_id: str, 
    new_name: str = Form(None),
    language: str = Form(None),
    region: str = Form(None), 
    description: str = Form(None),
    username: str = Depends(authenticate)
):
    """Rename a voice or update its metadata."""
    old_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    
    if not os.path.exists(old_path):
        raise HTTPException(status_code=404, detail="Voice not found")
        
    metadata = get_voice_metadata()
    if voice_id not in metadata:
        metadata[voice_id] = {"name": voice_id, "filename": f"{voice_id}.wav"}
        
    # Update fields
    if language: metadata[voice_id]["language"] = language
    if region: metadata[voice_id]["region"] = region
    if description: metadata[voice_id]["description"] = description
    
    # Handle rename
    if new_name and new_name != voice_id:
        safe_new_name = "".join([c for c in new_name if c.isalnum() or c in (" ", "-", "_")]).strip()
        new_path = os.path.join(VOICES_DIR, f"{safe_new_name}.wav")
        
        if os.path.exists(new_path):
             raise HTTPException(status_code=400, detail=f"Voice name '{safe_new_name}' already exists")
             
        os.rename(old_path, new_path)
        
        # Move metadata to new key
        metadata[safe_new_name] = metadata.pop(voice_id)
        metadata[safe_new_name]["name"] = safe_new_name
        metadata[safe_new_name]["filename"] = f"{safe_new_name}.wav"
        
    save_voice_metadata(metadata)
    
    return {"status": "Voice updated", "metadata": metadata.get(new_name if new_name else voice_id)}

@app.get("/download/{filename}")
async def download_file(filename: str, username: str = Depends(authenticate)):
    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.post("/demo")
async def demo_tts(
    text: str = Form(...),
    language: str = Form(None),
    region: str = Form(None)
):
    """
    Quick demo endpoint. 
    If language/region is provided, it attempts to find a matching voice.
    Otherwise defaults to female_english.
    """
    m = get_model()
    
    # Find best voice based on criteria
    voice_id = "female_english" # Default fallback
    
    if language or region:
        metadata = get_voice_metadata()
        best_match = None
        
        for vid, data in metadata.items():
            # Check for file existence first
            if not os.path.exists(os.path.join(VOICES_DIR, data.get("filename", f"{vid}.wav"))):
                continue
                
            score = 0
            if language and data.get("language") == language:
                score += 2
            if region and data.get("region") == region:
                score += 1
            
            # Simple scoring: prefer exact language+region (3), then language (2), then region (1)
            if score > 0:
                if best_match is None or score > best_match["score"]:
                    best_match = {"id": vid, "score": score}
        
        if best_match:
            voice_id = best_match["id"]

    voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    
    if not os.path.exists(voice_path):
        raise HTTPException(
            status_code=503, 
            detail=f"Voice '{voice_id}' not available. Please ensure voices are downloaded."
        )
    
    try:
        # Generate speech using the selected voice
        # Pass language if provided, otherwise default to "en"
        target_lang = language if language else "en"
        wav = m.generate(text, audio_prompt_path=voice_path, language=target_lang)
        
        out_id = str(uuid.uuid4())
        out_filename = f"demo_{out_id}.wav"
        out_path = os.path.join(STORAGE_DIR, out_filename)
        
        ta.save(out_path, wav, m.sr)
        
        # Save to history
        history_entry = {
            "id": out_id,
            "text": text,
            "filename": out_filename,
            "voice_used": voice_id,
            "language_requested": language, 
            "region_requested": region,
            "timestamp": datetime.now().isoformat(),
            "user": "anonymous"
        }
        save_to_history(history_entry)
        
        return FileResponse(out_path, media_type="audio/wav", filename=f"demo_{voice_id}.wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Voice Backend Running", "model": "ResembleAI/chatterbox-turbo", "auth": "enabled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
