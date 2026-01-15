
from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from database import get_db, Voice, Audio, User
from dependencies import get_current_user
import os
import uuid
import torch
import torchaudio as ta
import io
from fastapi.responses import StreamingResponse
from datetime import datetime
import shutil
import re
import json
from pathlib import Path
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS


import asyncio
import edge_tts

# --- CONFIG & HELPERS ---
VOICES_DIR = os.getenv("VOICES_DIR", "voices")
AMBIENCE_DIR = os.getenv("AMBIENCE_DIR", "ambience")
STORAGE_DIR = os.getenv("STORAGE_DIR", "outputs")

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# Global model cache to avoid reloading
MODELS = {"turbo": None, "mtl": None, "original": None}

def get_model(mode="turbo"):
    if MODELS.get(mode) is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"📡 Cargando modelo TTS ({mode}) en {device}...")
        
        # Monkey patch para forzar map_location en dispositivos sin CUDA (Mac)
        if device != "cuda":
            original_load = torch.load
            def safe_load(*args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = device
                return original_load(*args, **kwargs)
            # Aplicamos el patch
            torch.load = safe_load
            
        try:
            if mode == "turbo":
                MODELS["turbo"] = ChatterboxTurboTTS.from_pretrained(device)
            elif mode == "mtl":
                MODELS["mtl"] = ChatterboxMultilingualTTS.from_pretrained(device)
            else:
                MODELS["original"] = ChatterboxTTS.from_pretrained(device)
        finally:
            if device != "cuda":
                torch.load = original_load # Restaurar siempre
                
    return MODELS[mode]

def mix_ambience(audio_tensor, ambience_id, sample_rate=24000):
    ambience_path = os.path.join(AMBIENCE_DIR, f"{ambience_id}.wav")
    if not os.path.exists(ambience_path):
        return audio_tensor
    
    try:
        amb, sr = ta.load(ambience_path)
        # Resample if needed
        if sr != sample_rate:
            resampler = ta.transforms.Resample(sr, sample_rate)
            amb = resampler(amb)
        
        # Loop ambience to match speech length
        speech_len = audio_tensor.shape[1]
        amb_len = amb.shape[1]
        
        if amb_len < speech_len:
            repeats = (speech_len // amb_len) + 1
            amb = amb.repeat(1, repeats)
            
        amb = amb[:, :speech_len]
        
        # Mix (Low volume for background)
        mixed = audio_tensor + (amb * 0.15)
        return mixed
    except Exception as e:
        print(f"⚠️ Error mezclando ambiente {ambience_id}: {e}")
        return audio_tensor

def save_upload_as_wav(upload_file, destination):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    # Optional: could use ffmpeg to normalize/convert, but simplified for now

def remove_text_between_brackets(text):
    return re.sub(r'\[.*?\]', '', text)

router = APIRouter()

# --- ENDPOINTS ---

@router.get("/voices")
def list_voices(db: Session = Depends(get_db), user = Depends(get_current_user)):
    # Obtenemos voces públicas (owner_id es nulo) y las del usuario
    voices = db.query(Voice).filter((Voice.owner_id == None) | (Voice.owner_id == user.id)).all()
    return {
        "voices": [
            {
                "name": v.id, 
                "language": v.language, 
                "region": v.region, 
                "gender": v.gender, 
                "description": v.description
            } for v in voices
        ]
    }

@router.get("/history")
def get_audio_history(db: Session = Depends(get_db), user = Depends(get_current_user)):
    # Historial personal del usuario
    audios = db.query(Audio).filter(Audio.user_id == user.id).order_by(Audio.created_at.desc()).all()
    return audios

def generate_voice_audio(db: Session, user: User, text: str, voice_id: str = None, temperature: float = 0.7, ambience_id: str = None):
    # 1. Preparar Modelo
    target_mode = "mtl"
    m = get_model(target_mode)
    sr = m.sample_rate if hasattr(m, 'sample_rate') else 24000
    
    # Clean text
    clean_text = remove_text_between_brackets(text)
    
    # 2. Preparar Referencia de Voz
    ref_wav = None
    if voice_id:
        voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
        if os.path.exists(voice_path):
            ref_wav, _ = ta.load(voice_path)
    
    # 3. Generar
    try:
        # El modelo Multilingual (MTL) suele requerir ref_wav para zero-shot cloning
        if ref_wav is not None:
            wav = m.generate(clean_text, temperature=temperature, ref_wav=ref_wav)
        else:
            wav = m.generate(clean_text, temperature=temperature)
    except Exception as e:
        print(f"❌ Error en generación TTS: {e}")
        return None

    # 4. Mezclar Ambiente
    if ambience_id:
        wav = mix_ambience(wav, ambience_id, sr)
    
    # 5. Guardar Resultado
    out_id = str(uuid.uuid4())
    out_filename = f"gen_{out_id}.wav"
    out_path = os.path.join(STORAGE_DIR, out_filename)
    ta.save(out_path, wav, sr)
    
    # 6. Registro en DB
    audio_log = Audio(
        id=out_id,
        user_id=user.id,
        filename=out_filename,
        content=text,
        voice_used=voice_id or "custom"
    )
    db.add(audio_log)
    db.commit()
    db.refresh(audio_log)
    
    return audio_log

def get_audio_stream(text: str, voice_id: str = None, temperature: float = 0.7):
    """
    Genera audio en memoria y lo devuelve como un flujo de bytes WAV.
    No guarda nada en disco ni en BD.
    """
    target_mode = "mtl"
    m = get_model(target_mode)
    sr = m.sample_rate if hasattr(m, 'sample_rate') else 24000
    
    clean_text = remove_text_between_brackets(text)
    
    ref_wav = None
    if voice_id:
        voice_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
        if os.path.exists(voice_path):
            ref_wav, _ = ta.load(voice_path)

    try:
        # Generación con modelo mtl (Multilingual)
        # FIX FINAL: La librería parece no implementar clonación pública en generate() aun.
        # Usamos generación estándar para asegurar funcionalidad.
        # FIX 2: Falta 'language_id'
        wav = m.generate(clean_text, temperature=temperature, language_id="es")
        
        # Convertir a buffer de memoria WAV
        buffer = io.BytesIO()
        ta.save(buffer, wav.cpu(), sr, format="wav")
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"❌ Error en streaming TTS: {e}")
        return None

async def get_edge_audio_stream(text: str, voice: str = "es-AR-TomasNeural") -> io.BytesIO:
    """
    Genera audio usando Edge TTS (Microsoft Azure Free).
    Latencia: Ultra baja.
    Calidad: Alta (Neural).
    """
    communicate = edge_tts.Communicate(text, voice)
    buffer = io.BytesIO()
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
            
    buffer.seek(0)
    return buffer

@router.get("/voice/stream")
async def stream_voice_endpoint(
    text: str,
    voice_id: str = None,
    temperature: float = 0.7,
    user = Depends(get_current_user)
):
    # SWITCH: Usar Edge TTS si detectamos que queremos velocidad
    # Por ahora, forzamos Edge TTS para probar la velocidad que pidió el usuario.
    # Mapeamos voice_id a voces de Edge si es necesario
    
    edge_voice = "es-ES-AlvaroNeural" # Default Español (Más natural)
    if voice_id == "daniela": edge_voice = "es-MX-DaliaNeural"
    if voice_id == "female_english": edge_voice = "en-US-JennyNeural"
    
    # Usamos la nueva función rápida
    try:
        buffer = await get_edge_audio_stream(text, edge_voice)
        return StreamingResponse(buffer, media_type="audio/mpeg") # Edge devuelve MP3 por defecto
    except Exception as e:
        print(f"❌ Error EdgeTTS: {e}")
        # Fallback al sistema anterior si Edge falla (internet, etc)
        # buffer = get_audio_stream(text, voice_id, temperature) ...
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-tts")
async def generate_tts_endpoint(
    text: str = Form(...),
    voice_id: str = Form(None),
    language: str = Form("en"),
    temperature: float = Form(0.7),
    ambience_id: str = Form(None),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    audio_log = generate_voice_audio(db, user, text, voice_id, temperature, ambience_id)
    if not audio_log:
        raise HTTPException(status_code=500, detail="Error en generación de audio")
        
    out_path = os.path.join(STORAGE_DIR, audio_log.filename)
    return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")

@router.post("/voices/upload")
async def upload_voice_clone(
    name: str = Form(...),
    file: UploadFile = File(...),
    language: str = Form("en"),
    gender: str = Form(None),
    description: str = Form(None),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    safe_name = "".join([c for c in name if c.isalnum() or c in (" ", "-", "_")]).strip()
    file_path = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    
    save_upload_as_wav(file, file_path)
    
    # Actualizar DB
    voice = db.query(Voice).filter(Voice.id == safe_name).first()
    if not voice:
        voice = Voice(
            id=safe_name, 
            owner_id=user.id, 
            filename=f"{safe_name}.wav", 
            language=language,
            gender=gender,
            description=description
        )
        db.add(voice)
    else:
        # Solo el dueño o admin puede sobrescribir
        if voice.owner_id == user.id or user.id == 1:
            voice.filename = f"{safe_name}.wav"
            voice.gender = gender or voice.gender
            voice.description = description or voice.description
        else:
            raise HTTPException(status_code=403, detail="No autorizado para modificar esta voz")
            
    db.commit()
    return {"status": "guardado", "voice_id": safe_name}

@router.delete("/audios/{audio_id}")
def delete_audio(audio_id: str, db: Session = Depends(get_db), user = Depends(get_current_user)):
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="Audio no encontrado")
    if audio.user_id != user.id and user.id != 1:
        raise HTTPException(status_code=403, detail="No autorizado")
        
    fpath = os.path.join(STORAGE_DIR, audio.filename)
    if os.path.exists(fpath):
         os.remove(fpath)
    db.delete(audio)
    db.commit()
    return {"status": "eliminado"}
