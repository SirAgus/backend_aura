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

import sys
import types
from unittest.mock import MagicMock

# Mock xformers before importing audiocraft
# We use ModuleType and provide a __spec__ to satisfy libraries like 'diffusers'
def mock_module(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return m

import importlib.machinery
xformers_mock = mock_module("xformers")
xformers_mock.__version__ = "0.0.22"
xformers_ops_mock = mock_module("xformers.ops")
xformers_ops_mock.unbind = torch.unbind
# Frequently checked attributes
xformers_ops_mock.fmha = MagicMock()
xformers_ops_mock.memory_efficient_attention = MagicMock()
xformers_ops_mock.LowerTriangularMask = MagicMock()
xformers_mock.ops = xformers_ops_mock

# Add info module and version check for deeper validations
xformers_info_mock = mock_module("xformers.info")
xformers_info_mock.get_xformers_version = lambda: "0.0.22"
xformers_mock.info = xformers_info_mock

# Mock fairinternal checkpoint
xformers_checkpoint_mock = mock_module("xformers.checkpoint_fairinternal")
xformers_checkpoint_mock.checkpoint = MagicMock()
xformers_checkpoint_mock._get_default_policy = MagicMock()

sys.modules["xformers"] = xformers_mock
sys.modules["xformers.ops"] = xformers_ops_mock
sys.modules["xformers.info"] = xformers_info_mock
sys.modules["xformers.checkpoint_fairinternal"] = xformers_checkpoint_mock

try:
    from audiocraft.models import AudioGen
    print("✅ Audiocraft AudioGen support loaded")
except ImportError:
    print("⚠️ Audiocraft not found or could not be loaded. Dynamic ambience disabled.")
    AudioGen = None

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
    
    # Using robust URLs (High Quality Samples from SoundJay & Pixabay - Royalty Free)
    reliable_sources = {
        "rain": "https://www.soundjay.com/nature/sounds/rain-03.mp3",
        "birds": "https://www.soundjay.com/nature/sounds/canary-singing-01.mp3", 
        "storm": "https://www.soundjay.com/nature/sounds/thunder-01.mp3",
        "office": "https://www.soundjay.com/misc/sounds/busy-office-ambience-1.mp3",
        "forest": "https://www.soundjay.com/nature/sounds/forest-ambience-1.mp3", 
        "wind": "https://www.soundjay.com/nature/sounds/wind-howl-01.mp3",
        "cafe": "https://www.soundjay.com/misc/sounds/restaurant-1.mp3",
        "lofi": "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3" # Short chill loop
    }

    headers = {"User-Agent": "Mozilla/5.0"}

    for name, url in reliable_sources.items():
        fname = amb_path / f"{name}.wav"
        if not fname.exists():
            try:
                print(f"  - Downloading ambience: {name}...")
                response = requests.get(url, timeout=15, headers=headers)
                if response.status_code == 200:
                    temp = amb_path / f"temp_{name}.mp3"
                    with open(temp, "wb") as f:
                        f.write(response.content)
                    
                    # Robust load using torchaudio (handles mp3 if ffmpeg is present)
                    try:
                        w, sr = ta.load(str(temp))
                        ta.save(str(fname), w, sr)
                        print(f"    ✓ Ambient sound ready: {name}")
                    except Exception as load_err:
                        print(f"    ⚠️ Downloaded but failed to decode {name}: {load_err}")
                    finally:
                        if os.path.exists(temp):
                            os.remove(temp)
                else:
                    print(f"    ⚠️ URL failed for {name} (HTTP {response.status_code})")
            except Exception as e:
                print(f"    ✗ Download failed for {name}: {e}")
                
            # If still missing, generate simple fallback
            if not fname.exists():
                print(f"    generating fallback noise for {name}")
                dur = 10
                sr = 24000
                noise = torch.randn(1, dur * sr) * 0.05
                ta.save(str(fname), noise, sr)

# Execute on startup
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
audiogen_model = None

# --- LLM TEXT GENERATION (Qwen 2.5 3B) ---
llm_model = None
llm_tokenizer = None
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def get_llm():
    global llm_model, llm_tokenizer
    if llm_model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"🧠 Loading LLM: {LLM_MODEL_ID}...")
            # Use MPS (Mac) or CUDA if available
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load Tokenizer
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            
            # Load Model (Optimized for inference)
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID, 
                torch_dtype=torch.float16, # Use fp16 for speed/memory on MPS
                device_map=device
            )
            print(f"✅ LLM loaded on {device}!")
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            return None, None
            
    return llm_model, llm_tokenizer

@app.post("/chat")
async def chat_generation(
    prompt: str = Form(...),
    system_prompt: str = Form("You are a helpful assistant."),
    max_tokens: int = Form(200),
    temperature: float = Form(0.7),
    username: str = Depends(authenticate)
):
    """
    Generates text using the local lightweight LLM (3B).
    Useful for creating content to be spoken by the TTS.
    """
    model, tokenizer = get_llm()
    if not model:
        raise HTTPException(status_code=500, detail="LLM model could not be loaded.")
        
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Format input using chat template
        text_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {"response": response_text, "model": LLM_MODEL_ID}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

def get_audiogen():
    global audiogen_model

    if AudioGen is None:
        return None
    if audiogen_model is None:
        # Use MPS if available for Apple Silicon M4, else CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🚀 Loading AudioGen model on {device}...")
        try:
            # AudioGen-Medium is ~1.5B parameters. M4 handles it well on MPS.
            audiogen_model = AudioGen.get_pretrained('facebook/audiogen-medium', device=device)
            print(f"✅ AudioGen loaded and optimized for {device}!")
        except Exception as e:
            print(f"⚠️ Failed to load AudioGen on {device}: {e}")
            if device == "mps":
                print("🔄 Falling back to CPU for AudioGen...")
                try:
                    audiogen_model = AudioGen.get_pretrained('facebook/audiogen-medium', device="cpu")
                    print("✅ AudioGen loaded on CPU (slower but safe).")
                except Exception as e2:
                    print(f"❌ Failed to load AudioGen even on CPU: {e2}")
                    return None
            else:
                return None
    return audiogen_model

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
    # Removed brown noise generator
    pass
def process_text_tags(text, current_params, mode="turbo"):
    """
    Parses tags and identifies context.
    Returns: (modified_text, modified_params, auto_ambience)
    """
    import re
    
    # 1. Automatic Ambience Context Detection (if no tags are used)
    auto_ambience = None
    context_keywords = {
        "rain": ["lluvia", "llueve", "lloviendo", "tormenta", "rain", "storm"],
        "beach": ["playa", "mar", "olas", "oceano", "beach", "ocean", "waves"],
        "forest": ["bosque", "selva", "arboles", "naturaleza", "forest", "jungle", "woods"],
        "birds": ["pajaros", "aves", "canto", "birds", "chirping"],
        "cafe": ["cafe", "cafeteria", "barista", "restaurant", "coffe"],
        "office": ["oficina", "trabajo", "tecleando", "office", "typing", "work"],
        "lofi": ["estudiar", "relajo", "musica suave", "lofi", "study", "relax"],
        "storm": ["tormenta", "truenos", "relampagos", "storm", "thunder", "lightning"],
        "wind": ["viento", "brisa", "wind", "breeze"],
        "static": ["estatica", "ruido", "static", "noise"],
    }
    
    for amb_id, keywords in context_keywords.items():
        if amb_id in VALID_AMBIENCES and any(re.search(r'\b' + kw + r'\b', text, re.IGNORECASE) for kw in keywords):
            auto_ambience = amb_id
            break

    # 2. Onomatopoeias (Sound Effects)
    replacements = {
        r"\[risa\]": " [laugh] ",
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
        r"\[pausa\]": " ... ",
        r"\[beso\]": " mua ",
    }
    
    has_laughter = False
    for pattern, replacement in replacements.items():
        if re.search(pattern, text, re.IGNORECASE):
            if "laugh" in replacement or "chuckle" in replacement:
                has_laughter = True
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Dynamic Param Tuning
    if has_laughter or "[laugh]" in text or "[chuckle]" in text:
        current_params["temperature"] = max(current_params.get("temperature", 0.7), 1.1)
        current_params["exaggeration"] = max(current_params.get("exaggeration", 0.5), 1.2)
        current_params["repetition_penalty"] = 1.05
        print("🎭 Laughter/Expressivity detected! Tuning params.")

    # 4. Tone Tags
    tone_map = {
        r"\[(feliz|happy)\]": {"exaggeration": 1.5, "temperature": 1.0},
        r"\[(triste|sad)\]": {"exaggeration": 0.2, "temperature": 0.6, "cfg_weight": 0.9},
        r"\[(serio|serious)\]": {"exaggeration": 0.4, "temperature": 0.5, "cfg_weight": 1.0},
    }
    
    for pattern, p_updates in tone_map.items():
        if re.search(pattern, text, re.IGNORECASE):
            current_params.update(p_updates)
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 5. Model-Specific Cleaning
    if mode == "multilingual":
        para_tags = ["laugh", "sigh", "cough", "clear throat", "shush", "groan", "sniff", "gasp", "chuckle"]
        for p_tag in para_tags:
            text = text.replace(f"[{p_tag}]", "")
            text = text.replace(f" [{p_tag}] ", " ")

    return text.strip(), current_params, auto_ambience

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
    min_p: float = Form(0.05), # LOWERED to 0.05 for more expressiveness by default
    repetition_penalty: float = Form(2.0),
    top_p: float = Form(1.0),
    ambience_id: str = Form(None), # Predefined Ambience ID
    ambience_prompt: str = Form(None), # Custom Ambience Prompt
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
    processed_text_val, params, auto_amb = process_text_tags(text, params, actual_mode)
    
    # Apply auto ambience only if none is selected
    if not ambience_id and not ambience_prompt and auto_amb:
        print(f"🧠 [Auto-Context] Detected keyword, using ambience: {auto_amb}")
        ambience_id = auto_amb
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
        # --- DYNAMIC AMBIENCE TAG PARSING ---
        # We look for [ambience:something] or predefined tags like [rain], [birds], etc.
        active_ambience_id = ambience_id
        active_ambience_prompt = ambience_prompt
        
        # regex to find [ambience:...] or just common tags, with optional duration
        # We'll split the text into segments
        import re
        # IDs join
        predefined_ids = ["rain", "birds", "forest", "beach", "storm", "office", "cafe", "lofi", "static", "fire", "wind", "ags"]
        ids_pattern = "|".join(predefined_ids)
        
        tag_pattern = r"\[ambience:([^\]]+)\]|\[(" + ids_pattern + r"(?::\d+s)?)\]"
        
        parts = re.split(tag_pattern, processed_text_val, flags=re.IGNORECASE)
        # re.split returns: [text, group1, group2, text, ...]
        
        segments = []
        current_text = ""
        
        # Initial state
        seg_amb_id = active_ambience_id
        seg_amb_prompt = active_ambience_prompt
        seg_min_duration = 0.0

        # Logic upgrade:
        # [tag] text [tag] text -> segments
        # 1. Split creates: "", tag, "", text, tag, "", text...
        # We iterate and update 'current_state' whenever we hit a tag.
        
        i = 0
        while i < len(parts):
            # parts structure from re.split with groups:
            # [text_chunk, group1(custom), group2(predefined), text_chunk, ...]
            
            chunk = parts[i] # This is always a text chunk
            if chunk and chunk.strip():
                segments.append({
                    "text": chunk.strip(),
                    "amb_id": seg_amb_id,
                    "amb_prompt": seg_amb_prompt,
                    "min_duration": seg_min_duration
                })
                seg_min_duration = 0.0 # reset duration request after applied
            
            # Check for next tag (if exists)
            if i + 1 < len(parts):
                tag_custom = parts[i+1] # [ambience:...]
                tag_predef = parts[i+2] # [rain]
                
                val = tag_custom if tag_custom else tag_predef
                
                if val:
                    # Parse duration suffix if present
                    duration_match = re.search(r":(\d+)s$", val)
                    new_duration = 0.0
                    if duration_match:
                        new_duration = float(duration_match.group(1))
                        val = val[:duration_match.start()]
                    
                    seg_min_duration = new_duration
                    
                    # Logic: If we encounter the SAME tag again, it acts as a "Stop/Toggle"? 
                    # User request: "[rain] hola [rain]" -> rain only inside.
                    # Or "[rain] hola" -> rain until end.
                    
                    # Simple state machine:
                    # Any tag sets the current ambience.
                    # If we hit the SAME tag that is currently active, we assume it's a "closing" tag -> set ambience to None
                    # If we hit a DIFFERENT tag, we switch to that ambience.
                    
                    current_active_id = seg_amb_id
                    new_id = val
                    
                    if new_id == current_active_id:
                        # Toggle OFF
                        seg_amb_id = None
                        seg_amb_prompt = None
                    else:
                        # Toggle ON / SWITCH
                        seg_amb_id = val
                        seg_amb_prompt = None if tag_predef else val
            
            i += 3 # Jump text+g1+g2

        # Fallback if no tags were found or split didn't result in segments
        if not segments:
            segments = [{"text": processed_text_val, "amb_id": active_ambience_id, "amb_prompt": active_ambience_prompt, "min_duration": 0.0}]

        final_wavs = []
        
        for seg in segments:
            txt = seg["text"]
            aid = seg["amb_id"]
            aprompt = seg["amb_prompt"]
            dur = seg.get("min_duration", 0.0)
            
            print(f"  🧵 Generating segment: '{txt[:20]}...' with Ambience: {aid or aprompt or 'None'} (Duration: {dur}s)")
            
            # SPLIT LONG TEXT within segment if needed
            chunks = re.split(r'(?<=[.!?])\s+', txt)
            chunks = [c.strip() for c in chunks if c.strip()] or [txt]
            
            segment_wav_parts = []
            for chunk in chunks:
                if actual_mode == "multilingual":
                    w = m.generate(chunk, audio_prompt_path=prompt_path, language_id=language_id, 
                                     temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                                     repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"])
                else:
                    w = m.generate(chunk, audio_prompt_path=prompt_path, 
                                     temperature=params["temperature"], exaggeration=params["exaggeration"], cfg_weight=params["cfg_weight"],
                                     repetition_penalty=params["repetition_penalty"], top_p=params["top_p"], min_p=params["min_p"])
                segment_wav_parts.append(w)
            
            seg_voice_wav = torch.cat(segment_wav_parts, dim=1)
            
            # Mix Ambience for this specific segment
            if aid or aprompt:
                # Use m.sample_rate if available, else default
                sr = m.sample_rate if hasattr(m, 'sample_rate') else 24000
                seg_voice_wav = mix_ambience(seg_voice_wav, aid, sr, aprompt, min_duration=dur)
            
            final_wavs.append(seg_voice_wav)

        # Concatenate all segments
        wav = torch.cat(final_wavs, dim=1)

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
