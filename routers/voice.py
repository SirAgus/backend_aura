
from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from database import get_db, Voice, Audio, User
from dependencies import get_current_user
import os
import uuid
import torch
import torchaudio as ta
from datetime import datetime
import shutil
import re
import json
from pathlib import Path
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS


# If import failed (as handled in previous step manually), we define local
def remove_text_between_brackets(text):
    return re.sub(r'\[.*?\]', '', text)

router = APIRouter()

# --- CONFIG ---
STORAGE_DIR = os.getenv("STORAGE_DIR", "outputs")

# --- ENDPOINTS ---

@router.post("/generate-tts")
async def generate_tts_endpoint(
    text: str = Form(...),
    voice_id: str = Form(None),
    audio_prompt: UploadFile = File(None),
    language: str = Form("en"),
    temperature: float = Form(0.7),
    ambience_id: str = Form(None),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    # 1. Prepare Model
    target_mode = "turbo" # default
    if voice_id:
         # Check if multilingual needed (simplified logic)
         pass 

    m = get_model(target_mode)
    
    # 2. Text Preprocessing
    processed_text_val = text
    
    # Basic generation loop
    wavs = []
    
    # Simple split if no complex tags
    chunks = [text] 
    
    for chunk in chunks:
        # Generate
        wav = m.generate(chunk, temperature=temperature) 
        
        # Mix Ambience (Global Form Param)
        if ambience_id:
            sr = m.sample_rate if hasattr(m, 'sample_rate') else 24000
            wav = mix_ambience(wav, ambience_id, sr)
            
        wavs.append(wav)
        
    final_wav = torch.cat(wavs, dim=1) if wavs else torch.zeros(1, 16000)
    
    # Save
    out_id = str(uuid.uuid4())
    out_filename = f"gen_{out_id}.wav"
    out_path = os.path.join(STORAGE_DIR, out_filename)
    ta.save(out_path, final_wav, m.sample_rate)
    
    # DB Log
    audio_log = Audio(
        id=out_id,
        user_id=user.id,
        filename=out_filename,
        content=text, # Save FULL text
        voice_used=voice_id or "custom"
    )
    db.add(audio_log)
    db.commit()
    
    return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")

@router.post("/voices/upload")
async def upload_voice_clone(
    name: str = Form(...),
    file: UploadFile = File(...),
    language: str = Form("en"),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    safe_name = "".join([c for c in name if c.isalnum() or c in (" ", "-", "_")]).strip()
    file_path = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    
    save_upload_as_wav(file, file_path)
    
    # DB Update
    voice = db.query(Voice).filter(Voice.id == safe_name).first()
    if not voice:
        voice = Voice(id=safe_name, owner_id=user.id, filename=f"{safe_name}.wav", language=language)
        db.add(voice)
    else:
        # Update existing? Only if owner
        if voice.owner_id == user.id or user.id == 1:
             voice.filename = f"{safe_name}.wav"
    db.commit()
    
    return {"status": "saved", "voice_id": safe_name}

@router.delete("/audios/{audio_id}")
def delete_audio(audio_id: str, db: Session = Depends(get_db), user = Depends(get_current_user)):
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")
    if audio.user_id != user.id and user.id != 1:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    fpath = os.path.join(STORAGE_DIR, audio.filename)
    if os.path.exists(fpath):
         os.remove(fpath)
    db.delete(audio)
    db.commit()
    return {"status": "deleted"}
