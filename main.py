from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os
import uuid
import torch
import torchaudio as ta
# TTS Models: ChatterboxTurboTTS and ChatterboxMultilingualTTS
print("ℹ️ Multi-model support enabled (Turbo and Multilingual)")

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

# Load the models (cached)
models = {"turbo": None, "multilingual": None}

def process_text_tags(text, current_params):
    """
    Parses tags like [risa], [happy] and modifies the text or parameters.
    Returns: (modified_text, modified_params)
    """
    import re
    
    # 1. Onomatopoeias (Sound Effects)
    replacements = {
        r"\[risa\]": " ¡jajaja! ",
        r"\[laugh\]": " haha! ",
        r"\[laught\]": " haha! ", # Typos support
        r"\[jaja\]": " ¡ja ja ja! ",
        r"\[suspiro\]": " ...aaaaay... ",
        r"\[sigh\]": " ...ahh... ",
        r"\[tos\]": " cof cof ",
        r"\[beso\]": " mua ",
        r"\[pausa\]": " ... ",
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Style Tags (Global adjustment heuristic)
    # Check if text starts with a style tag to adjust global parameters
    # Note: Only works if the tag is at the start or dominant
    
    if re.search(r"\[(feliz|happy)\]", text, re.IGNORECASE):
        # Happy = High exaggeration, higher pitch variation
        current_params["exaggeration"] = max(current_params.get("exaggeration", 0.5), 1.2)
        current_params["temperature"] = max(current_params.get("temperature", 0.8), 0.9)
        text = re.sub(r"\[(feliz|happy)\]", "", text, flags=re.IGNORECASE)
        
    if re.search(r"\[(triste|sad)\]", text, re.IGNORECASE):
        # Sad = Low exaggeration, slower, stable
        current_params["exaggeration"] = 0.2
        current_params["temperature"] = 0.6
        current_params["cfg_weight"] = 0.8 # More constraint
        text = re.sub(r"\[(triste|sad)\]", "", text, flags=re.IGNORECASE)

    if re.search(r"\[(serio|serious)\]", text, re.IGNORECASE):
        # Serious = Low temp, neutral exaggeration
        current_params["exaggeration"] = 0.4
        current_params["temperature"] = 0.5
        current_params["cfg_weight"] = 0.9
        text = re.sub(r"\[(serio|serious)\]", "", text, flags=re.IGNORECASE)

    return text.strip(), current_params

def get_model(mode="turbo"):
    global models
    
    # Sanitize mode
    if mode not in ["turbo", "multilingual"]:
        mode = "turbo"
        
    if models[mode] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check for Apple Silicon (MPS)
        if device == "cpu" and torch.backends.mps.is_available():
            # Chatterbox is better on CPU for now to avoid compatibility issues
            pass

        if mode == "multilingual":
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                print(f"Loading ChatterboxMultilingualTTS on {device}...")
                
                # Monkeypatch torch.load to fix the "deserialize on CUDA device" bug in the library
                orig_load = torch.load
                if device == "cpu":
                    def patched_load(*args, **kwargs):
                        if 'map_location' not in kwargs:
                            kwargs['map_location'] = 'cpu'
                        return orig_load(*args, **kwargs)
                    torch.load = patched_load
                
                try:
                    models["multilingual"] = ChatterboxMultilingualTTS.from_pretrained(device=device)
                    print("✅ Multilingual model loaded successfully!")
                finally:
                    # Restore original torch.load
                    if device == "cpu":
                        torch.load = orig_load
                        
            except Exception as e:
                print(f"⚠️ Failed to load Multilingual model: {e}")
                print("🔄 Falling back to Turbo model...")
                # If multilingual fails, we must force the return of turbo
                return get_model("turbo")
        else:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            print(f"Loading ChatterboxTurboTTS on {device}...")
            models["turbo"] = ChatterboxTurboTTS.from_pretrained(device=device)
            
    return models[mode]



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
    language: str = Form("en"), # Language for voice selection (reference sample)
    mode: str = Form("turbo"), # "turbo" or "multilingual"
    language_id: str = Form("es"), # Language for the model (for multilingual mode)
    temperature: float = Form(0.8), # Higher = more emotional/varied, Lower = more stable
    exaggeration: float = Form(0.5), # Higher = more expressive
    cfg: float = Form(0.6), # Slightly higher for more stability
    repetition_penalty: float = Form(1.15), # Balanced for Spanish
    top_p: float = Form(0.9), # Nucleus sampling. Lower = more "stable" voice qualities.
    min_p: float = Form(0.05), # Minimum probability filter.
    username: str = Depends(authenticate)
):
    m = get_model(mode)
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    actual_mode = "turbo" if isinstance(m, ChatterboxTurboTTS) else "multilingual"

    print(f"🚀 TTS Request: Mode [{actual_mode}], Language ID [{language_id}], Voice ID [{voice_id if voice_id else 'upload'}]")
    if actual_mode == "turbo" and language == "es":
        print("💡 Tip: Usando modo 'turbo' con español. Para mejor acento, usa mode='multilingual'.")



    
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
        # Support both 'Jorgete' and 'Jorgete.wav' styles
        clean_voice_id = voice_id.replace(".wav", "")
        voice_path = os.path.join(VOICES_DIR, f"{clean_voice_id}.wav")
        
        if os.path.exists(voice_path):
            prompt_path = voice_path
            print(f"✅ Using stored voice: {voice_path}")
        else:
            # Try to find by name in metadata if filename mismatch
            metadata = get_voice_metadata()
            if clean_voice_id in metadata:
                v_filename = metadata[clean_voice_id].get("filename")
                voice_path = os.path.join(VOICES_DIR, v_filename)
                if os.path.exists(voice_path):
                    prompt_path = voice_path
                    print(f"✅ Using stored voice (from metadata): {voice_path}")
            
            if not prompt_path:
                raise HTTPException(status_code=404, detail=f"Voice clone '{voice_id}' not found in local storage.")
    
    try:
        if prompt_path:
            # Generate speech using the reference voice (cloning)
            if actual_mode == "multilingual":
                # Multilingual model parameters
                wav = m.generate(processed_text, audio_prompt_path=prompt_path, language_id=language_id, 
                                 temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                                 repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"])
            else:
                # Turbo model parameters
                wav = m.generate(processed_text, audio_prompt_path=prompt_path, 
                                 temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                                 repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"])
        else:
            return {"error": "audio_prompt or voice_id is required for Chatterbox"}



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
            "mode": actual_mode,
            "language_id": language_id if actual_mode == "multilingual" else "en",
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
    new_name: str = None,
    language: str = None,
    region: str = None, 
    description: str = None,
    username: str = Depends(authenticate)
):
    """Rename a voice or update its metadata. Accepts Query Params."""
    # Debug print
    print(f"Updating voice: {voice_id}")
    
    # Try to find file with exact match first
    old_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    
    if not os.path.exists(old_path):
        # Try finding it URL decoded just in case
        import urllib.parse
        decoded_id = urllib.parse.unquote(voice_id)
        old_path = os.path.join(VOICES_DIR, f"{decoded_id}.wav")
        if not os.path.exists(old_path):
            print(f"File not found at: {old_path}")
            raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
        voice_id = decoded_id
        
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


@app.api_route("/demo", methods=["GET", "POST"])
async def demo_tts(request: Request):
    """
    Quick demo endpoint. Accepts GET or POST.
    Robustly handles parameters from either Query Params (GET) or Form Data (POST).
    """
    # Extract parameters manually depending on method
    if request.method == "GET":
        language = request.query_params.get("language")
        region = request.query_params.get("region")
        mode = request.query_params.get("mode", "turbo")
        voice_id_req = request.query_params.get("voice_id")
        language_id = request.query_params.get("language_id", "es")
        temperature = float(request.query_params.get("temperature", 0.8))
        exaggeration = float(request.query_params.get("exaggeration", 0.5))
        cfgValue = float(request.query_params.get("cfg", 0.5))
        rep_p = float(request.query_params.get("repetition_penalty", 1.2))
        t_p = float(request.query_params.get("top_p", 0.9))
    else: # POST
        form_data = await request.form()
        text = form_data.get("text", "This is a voice cloning demo.")
        language = form_data.get("language")
        region = form_data.get("region")
        mode = form_data.get("mode", "turbo")
        voice_id_req = form_data.get("voice_id")
        language_id = form_data.get("language_id", "es")
        temperature = float(form_data.get("temperature", 0.8))
        exaggeration = float(form_data.get("exaggeration", 0.5))
        cfgValue = float(form_data.get("cfg", 0.5))
        rep_p = float(form_data.get("repetition_penalty", 1.2))
        t_p = float(form_data.get("top_p", 0.9))

    m = get_model(mode)
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    actual_mode = "turbo" if isinstance(m, ChatterboxTurboTTS) else "multilingual"

    print(f"🚀 Demo Request: Mode [{actual_mode}], Language [{language}], Language ID [{language_id}]")
    if actual_mode == "turbo" and language == "es":
        print("💡 Tip: Demo en español usando modo 'turbo'. Prueba con mode='multilingual' para acento nativo.")


    
    # Find best voice based on criteria
    voice_id = "female_english" # Default fallback
    
    if voice_id_req:
        voice_id = voice_id_req
    elif language or region:
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
        # Fallback to absolute default if selected voice is missing
        voice_path = os.path.join(VOICES_DIR, "female_english.wav")
        if not os.path.exists(voice_path):
             raise HTTPException(
                status_code=503, 
                detail="No suitable voice found. Please restart server to download defaults."
            )
    
    try:
        # Generate speech using the selected voice
        # The language is determined by the reference voice, not by a language parameter
        # The language parameter is only used for voice selection

        # Generate speech using the selected voice
        # Turbo model determines language by reference voice
        if actual_mode == "multilingual":
            wav = m.generate(processed_text, audio_prompt_path=voice_path, language_id=language_id, 
                             temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                             repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"] if "min_p" in params else 0.05)
        else:
            wav = m.generate(processed_text, audio_prompt_path=voice_path, 
                             temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                             repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"] if "min_p" in params else 0.05)



        # Convert to bytes for streaming response (don't save to disk)
        import io
        buffer = io.BytesIO()
        ta.save(buffer, wav, m.sr, format="wav")
        buffer.seek(0)

        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=demo_{voice_id}.wav"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/")
def read_root():
    return {
        "status": "Voice Backend Running", 
        "model": "ResembleAI/chatterbox-turbo", 
        "multilingual_support": "Enabled",
        "available_modes": ["turbo", "multilingual"],
        "auth": "enabled"
    }


@app.post("/login")
async def login(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify user credentials.
    Returns success/failure without exposing sensitive information.
    """
    if credentials.username == API_USERNAME and credentials.password == API_PASSWORD:
        return {
            "success": True,
            "message": "Login successful",
            "user": credentials.username
        }
    else:
        return {
            "success": False,
            "message": "Invalid credentials"
        }

@app.get("/voices/list")
async def list_available_voices():
    """
    List all available voices with their metadata.
    Use /demo endpoint to test them with your own text.
    """
    if not os.path.exists(VOICES_DIR):
        return {"voices": []}

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
            "gender": None,
            "description": "Voice available"
        })
        # Ensure filename matches existing file
        meta["filename"] = f
        meta["id"] = voice_id
        meta["preview_url"] = f"/demo?text=Hola%20mundo&voice_id={voice_id}"
        result.append(meta)

    return {"voices": result, "total": len(result)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
