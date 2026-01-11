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
AMBIENCE_DIR = os.getenv("AMBIENCE_DIR", "ambience")

# Ensure local directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(AMBIENCE_DIR, exist_ok=True)

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

# Default ambience sounds
DEFAULT_AMBIENCE = {
    "rain": "https://pixabay.com/sound-effects/download/heavy-rain-nature-sounds-8167.mp3", # Example public domain/royalty free or placeholder
    "birds": "https://pixabay.com/sound-effects/download/birds-singing-03-6756.mp3",
    "cafe": "https://pixabay.com/sound-effects/download/coffee-shop-chatter-8178.mp3",
    "lofi": "https://pixabay.com/sound-effects/download/lofi-study-112191.mp3"
}

def download_default_ambience():
    """Download default ambience sounds if they don't exist."""
    amb_path = Path(AMBIENCE_DIR)
    print("Checking default ambience sounds...")
    
    # We need to simulate headers for some sites or use direct raw links. 
    # For now, we will use mock-up logic or direct valid URLs if available.
    # Note: Pixabay requires specific headers/auth often, so these are illustrative placeholders 
    # that would need valid direct raw URLs in a real prod env.
    
    # Using robust URLs
    reliable_sources = {
        "rain": "https://www.soundjay.com/nature/sounds/rain-03.mp3",
        "birds": "https://www.soundjay.com/nature/sounds/canary-singing-01.mp3",
        "storm": "https://www.soundjay.com/nature/sounds/thunder-01.mp3",
        "office": "https://www.soundjay.com/misc/sounds/busy-office-ambience-1.mp3"
    }

    headers = {"User-Agent": "Mozilla/5.0"}

    for name, url in reliable_sources.items():
        fname = amb_path / f"{name}.wav"
        if not fname.exists():
            try:
                print(f"  - Downloading ambience: {name}...")
                response = requests.get(url, timeout=15, headers=headers)
                if response.status_code == 200:
                    temp = amb_path / f"temp_{name}.ogg"
                    with open(temp, "wb") as f:
                        f.write(response.content)
                    w, sr = ta.load(str(temp))
                    ta.save(str(fname), w, sr)
                    os.remove(temp)
                    print(f"    ✓ Ambient sound ready: {name}")
                else:
                    raise Exception(f"HTTP {response.status_code}")
            except Exception as e:
                print(f"    ✗ Fallback: Generating noise for {name} ({e})")
                # Generate a colored noise as fallback
                dur = 10
                sr = 24000
                noise = torch.randn(1, dur * sr)
                if name == "birds":
                    # Soft chirps simulation or just silence is better than white noise
                    noise = noise * 0.05 
                elif name == "office":
                    # Mid range hum
                    noise = noise * 0.1
                else:
                    noise = noise * 0.05
                ta.save(str(fname), noise.real if torch.is_complex(noise) else noise, sr)

download_default_ambience()

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
models = {"turbo": None, "multilingual": None, "original": None}


def get_model(mode="turbo"):
    global models
    
    # Sanitize mode
    if mode not in ["turbo", "multilingual", "original"]:
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
        elif mode == "original":
            from chatterbox import ChatterboxTTS
            print(f"Loading Original ChatterboxTTS on {device}...")
            models["original"] = ChatterboxTTS.from_pretrained(device=device)
        else:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            print(f"Loading ChatterboxTurboTTS on {device}...")
            models["turbo"] = ChatterboxTurboTTS.from_pretrained(device=device)
            
    return models[mode]

def generate_brown_noise(duration_sec=30, sr=24000):
    """Generates a brown noise sample for testing ambience."""
    # Simple accumulation of white noise creates brown noise
    x = torch.randn(duration_sec * sr)
    brown = torch.cumsum(x, dim=0)
    # Normalize
    brown = brown / torch.max(torch.abs(brown))
    # Save
    path = os.path.join(AMBIENCE_DIR, "static_noise.wav")
    if not os.path.exists(path):
        ta.save(path, brown.unsqueeze(0), sr)
        print(f"✅ Generated sample ambience: {path}")

generate_brown_noise() # Run on startup

def mix_ambience(voice_wav, ambience_id, sr):
    """Mixes voice tensor with an ambience file."""
    if not ambience_id:
        return voice_wav
        
    print(f"DEBUG: mix_ambience called with id={ambience_id}, sr={sr}")
    amb_path = os.path.join(AMBIENCE_DIR, f"{ambience_id}.wav")
    if not os.path.exists(amb_path):
        print(f"⚠️ Ambience file not found: {amb_path}")
        return voice_wav

    try:
        # 1. Ensure voice_wav is a Tensor on CPU
        if not isinstance(voice_wav, torch.Tensor):
            if hasattr(voice_wav, '__array__'): # Numpy
                voice_wav = torch.from_numpy(voice_wav)
            elif isinstance(voice_wav, list):
                voice_wav = torch.tensor(voice_wav)
            else:
                print(f"⚠️ Unknown voice_wav type: {type(voice_wav)}")
                return voice_wav
        
        voice_wav = voice_wav.detach().cpu()
        if voice_wav.dim() == 1:
            voice_wav = voice_wav.unsqueeze(0) # [1, T]
            
        print(f"DEBUG: voice_wav shape: {voice_wav.shape}")

        # 2. Load Ambience
        amb_wav, amb_sr = ta.load(amb_path)
        amb_wav = amb_wav.detach().cpu()
        
        # 3. Resample
        if amb_sr != sr:
            resampler = ta.transforms.Resample(amb_sr, sr)
            amb_wav = resampler(amb_wav)
            
        # 4. Mono/Stereo Check & Normalization
        if voice_wav.shape[0] == 1 and amb_wav.shape[0] > 1:
            amb_wav = torch.mean(amb_wav, dim=0, keepdim=True)
            
        # NORMALIZE BOTH TO 1.0 (0dB) BEFORE MIXING
        v_max = torch.max(torch.abs(voice_wav))
        if v_max > 0:
            voice_wav = voice_wav / v_max
            
        a_max = torch.max(torch.abs(amb_wav))
        if a_max > 0:
            amb_wav = amb_wav / a_max
            
        print(f"DEBUG: Normalized voice (max={v_max:.2f}) and ambience (max={a_max:.2f})")
            
        # 5. Loop/Repeat or Seek Random Ambience
        voice_len = voice_wav.shape[1]
        amb_len = amb_wav.shape[1]
        
        import random
        if amb_len > voice_len:
            start_idx = random.randint(0, amb_len - voice_len)
            amb_wav = amb_wav[:, start_idx : start_idx + voice_len]
        else:
            repeats = (voice_len // amb_len) + 1
            amb_wav = amb_wav.repeat(1, repeats)
            amb_wav = amb_wav[:, :voice_len]
        
        print(f"DEBUG: amb_wav shape after processing: {amb_wav.shape}")

        # 7. Mix
        # Reduced ambience (0.25) to prevent overpowering the voice
        mixed = (voice_wav * 1.0) + (amb_wav * 0.25)
        
        # 8. Normalize result
        max_val = torch.max(torch.abs(mixed))
        if max_val > 0.01: # Avoid division by zero
            mixed = mixed / max_val
            
        return mixed

    except Exception as e:
        print(f"❌ Error mixing ambience: {e}")
        import traceback
        traceback.print_exc()
        # Return original audio if mixing fails
        return voice_wav

def process_text_tags(text, current_params):
    """
    Parses tags like [risa], [happy] and modifies the text or parameters.
    Returns: (modified_text, modified_params)
    """
    import re
    
    # 1. Onomatopoeias (Sound Effects)
    # Map Spanish tags to valid Model Tags (do NOT replace with text "haha", use the tag "[laugh]")
    replacements = {
        r"\[risa\]": " [laugh] ",
        r"\[laught\]": " [laugh] ", # Typos support
        r"\[jaja\]": " [laugh] ",
        r"\[broma\]": " [laugh] ",
        r"\[suspiro\]": " [sigh] ",
        r"\[tos\]": " [cough] ",
        r"\[carraspeo\]": " [clear throat] ",
        r"\[chistar\]": " [shush] ",
        r"\[quejido\]": " [groan] ",
        r"\[olfatear\]": " [sniff] ",
        r"\[jadeo\]": " [gasp] ",
        r"\[risita\]": " [chuckle] ",
        
        # Keep these as text if they aren't supported model tags
        r"\[pausa\]": " ... ",
        r"\[beso\]": " mua ",
    }
    
    has_laughter = False
    for pattern, replacement in replacements.items():
        if re.search(pattern, text, re.IGNORECASE):
            if "laugh" in replacement or "chuckle" in replacement:
                has_laughter = True
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Dynamic Param Tuning for Naturalness
    # If laughter is present, boost parameters to allow for more expressive/unstable outputs
    if has_laughter or "[laugh]" in text or "[chuckle]" in text:
        # Laughter needs high temperature to sound real (chaos)
        current_params["temperature"] = max(current_params.get("temperature", 0.7), 0.95)
        current_params["exaggeration"] = max(current_params.get("exaggeration", 0.5), 0.7)
        # Lower repetition penalty significantly for laughter, as it often involves rapid repetitive sounds
        current_params["repetition_penalty"] = 1.0 
        print("🎭 Laughter detected! Boosting temperature and lowering penalty for naturalness.")

    # 3. Tone Tags
    if re.search(r"\[(feliz|happy)\]", text, re.IGNORECASE):
        current_params["exaggeration"] = max(current_params.get("exaggeration", 0.5), 1.5)
        current_params["temperature"] = max(current_params.get("temperature", 0.8), 0.95)
        current_params["repetition_penalty"] = max(current_params.get("repetition_penalty", 1.1), 1.15)
        text = re.sub(r"\[(feliz|happy)\]", "", text, flags=re.IGNORECASE)
        
    if re.search(r"\[(triste|sad)\]", text, re.IGNORECASE):
        current_params["exaggeration"] = 0.2
        current_params["temperature"] = 0.6
        current_params["cfg_weight"] = 0.8
        text = re.sub(r"\[(triste|sad)\]", "", text, flags=re.IGNORECASE)

    if re.search(r"\[(serio|serious)\]", text, re.IGNORECASE):
        current_params["exaggeration"] = 0.4
        current_params["temperature"] = 0.5
        current_params["cfg_weight"] = 0.9
        text = re.sub(r"\[(serio|serious)\]", "", text, flags=re.IGNORECASE)

    return text.strip(), current_params

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
    repetition_penalty: float = Form(1.1), # LOWERED default from 1.15 to 1.1 for better flow
    top_p: float = Form(0.9), # Nucleus sampling
    min_p: float = Form(0.05), # LOWERED to 0.05 for more expressiveness by default
    ambience_id: str = Form(None), # New Ambience parameter
    username: str = Depends(authenticate)
):
    # Determine mode:
    # 1. If voice_id is provided, check its metadata to see if it has a preferred model.
    target_mode = mode
    if voice_id:
        try:
             metadata = get_voice_metadata()
             # handle both 'voice_id' and 'voice_id.wav' inputs
             clean_id = voice_id.replace(".wav", "")
             
             if clean_id in metadata:
                 voice_model = metadata[clean_id].get("model")
                 if voice_model == "chatterbox-multilingual":
                     target_mode = "multilingual"
                 elif voice_model == "chatterbox-turbo" or voice_model == "chatterbox-original":
                     target_mode = "turbo"
        except:
            pass

    m = get_model(target_mode)
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    actual_mode = "turbo" if isinstance(m, ChatterboxTurboTTS) else "multilingual"

    print(f"🚀 TTS Request: Mode [{actual_mode}], Ambience: [{ambience_id}], Text: {text[:30]}...")
    
    # Process text tags ONLY if using Turbo model (as per request specs)
    # Multilingual doesn't support paralinguistic tags in this implementation
    processed_text_val = text
    params = {
        "temperature": temperature,
        "exaggeration": exaggeration,
        "cfg_weight": cfg,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "min_p": min_p
    }
    
    # Process text tags for BOTH modes to allow dynamic param tuning
    # even if the model doesn't support the tags, removing them is good
    processed_text_val, params = process_text_tags(text, params)
    
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
        clean_voice_id = voice_id.replace(".wav", "")
        voice_path = os.path.join(VOICES_DIR, f"{clean_voice_id}.wav")
        
        if os.path.exists(voice_path):
            prompt_path = voice_path
        else:
            # Try to find by name in metadata if filename mismatch
            metadata = get_voice_metadata()
            if clean_voice_id in metadata:
                v_filename = metadata[clean_voice_id].get("filename")
                voice_path = os.path.join(VOICES_DIR, v_filename)
                if os.path.exists(voice_path):
                    prompt_path = voice_path
            
            if not prompt_path:
                raise HTTPException(status_code=404, detail=f"Voice clone '{voice_id}' not found in local storage.")
    
    try:
        if prompt_path:
            # Generate speech using the reference voice (cloning)
            
            # SPLIT LONG TEXT to prevent cut-off
            import re
            # Split but KEEP the delimiter to preserve punctuation/tone
            # This uses a lookbehind to keep . ! ?
            chunks = re.split(r'(?<=[.!?])\s+', processed_text_val)
            chunks = [c.strip() for c in chunks if c.strip()]
            
            # If re.split failed to find delimiters, we fall back to simple split or slice
            if len(chunks) == 1 and len(processed_text_val) > 150:
                 # Last resort: split by comma if no periods
                 chunks = re.split(r'(?<=,)\s+', processed_text_val)
                 chunks = [c.strip() for c in chunks if c.strip()]
            
            if len(chunks) > 1:
                print(f"📦 Splitting text into {len(chunks)} chunks...")
                chunk_wavs = []
                for i, chunk in enumerate(chunks):
                    print(f"  - Part {i+1}/{len(chunks)}: '{chunk[:30]}...'")
                    if actual_mode == "multilingual":
                        w = m.generate(chunk, audio_prompt_path=prompt_path, language_id=language_id, 
                                         temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                                         repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"])
                    else:
                        w = m.generate(chunk, audio_prompt_path=prompt_path, 
                                         temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                                         repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"])
                    chunk_wavs.append(w)
                
                # Concatenate waves
                wav = torch.cat(chunk_wavs, dim=1)
            else:
                # Normal generation for short text
                if actual_mode == "multilingual":
                    # Multilingual model parameters
                    wav = m.generate(processed_text_val, audio_prompt_path=prompt_path, language_id=language_id, 
                                     temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                                     repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"])
                else:
                    # Turbo model parameters
                    wav = m.generate(processed_text_val, audio_prompt_path=prompt_path, 
                                     temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                                     repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"])
        else:
            return {"error": "audio_prompt or voice_id is required for Chatterbox"}

        # Mix Ambience if requested
        if ambience_id:
             print(f"🔉 Mixing ambience: {ambience_id}")
             wav = mix_ambience(wav, ambience_id, m.sr)

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
    model: str = Form("chatterbox-turbo"),
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
            "model": model,
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
    model: str = None,
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
    if model: metadata[voice_id]["model"] = model
    
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
        text = request.query_params.get("text", "This is a voice cloning demo.")
        language = request.query_params.get("language")
        region = request.query_params.get("region")
        mode = request.query_params.get("mode", "turbo")
        voice_id_req = request.query_params.get("voice_id")
        language_id = request.query_params.get("language_id", "es")
        temperature = float(request.query_params.get("temperature", 0.8))
        exaggeration = float(request.query_params.get("exaggeration", 0.5))
        cfgValue = float(request.query_params.get("cfg", 0.6))
        rep_p = float(request.query_params.get("repetition_penalty", 1.15))
        t_p = float(request.query_params.get("top_p", 0.9))
        min_p = float(request.query_params.get("min_p", 0.05))
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
        cfgValue = float(form_data.get("cfg", 0.6))
        rep_p = float(form_data.get("repetition_penalty", 1.15))
        t_p = float(form_data.get("top_p", 0.9))
        min_p = float(form_data.get("min_p", 0.05))

    # Construct initial params dictionary
    initial_params = {
        "temperature": temperature,
        "exaggeration": exaggeration,
        "cfg_weight": cfgValue,
        "repetition_penalty": rep_p,
        "top_p": t_p,
        "min_p": min_p
    }

    # Process tags in text (this updates params based on tags like [happy])
    processed_text, params = process_text_tags(text, initial_params)

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
