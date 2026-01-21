
from fastapi import APIRouter, Depends, HTTPException, Form, Body, File, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db, Thread, Message
from dependencies import get_current_user
from pydantic import BaseModel
import threading
import queue
import json
import whisper
import tempfile
import shutil
import os
import re
import urllib.parse
import base64
import sys
import platform
from routers.voice import get_edge_audio_stream

router = APIRouter()

# --- LLM BACKEND SELECTION ---
USE_MLX = False
try:
    if sys.platform == "darwin" and platform.machine() == "arm64":
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler
        USE_MLX = True
        print("🍏 Detectado Chip Apple Silicon: Usando Backend MLX")
    else:
        raise ImportError("No es Apple Silicon")
except ImportError:
    print("🐧 Detectado Linux/Intel/CPU: Usando Backend Transformers (CPU/CUDA)")
    # Fallback libraries
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch

# --- MODELS CACHE ---
WHISPER_MODEL = None

def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("🎙️ Cargando modelo Whisper (STT)...")
        WHISPER_MODEL = whisper.load_model("base")
    return WHISPER_MODEL

# --- LLM SERVICE VARIABLE ---
llm_model = None
llm_tokenizer = None
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" # Modelo en HuggingFace Hub

def get_llm():
    global llm_model, llm_tokenizer
    
    if llm_model is not None:
        return llm_model, llm_tokenizer

    if USE_MLX:
        try:
            mlx_id = "mlx-community/Qwen2.5-3B-Instruct-4bit"
            print(f"🚀 Cargando LLM TURBO (MLX): {mlx_id}...")
            llm_model, llm_tokenizer = mlx_lm.load(mlx_id)
            print("✅ LLM optimizado para Mac cargado exitosamente!")
        except Exception as e:
            print(f"❌ Error al cargar MLX: {e}")
            return None, None
    else:
        try:
            print(f"⚙️ Cargando LLM Standard (Transformers CPU): {LLM_MODEL_ID}...")
            dtype = torch.float32 
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID, 
                torch_dtype=dtype,
                device_map="auto"
            )
            print("✅ LLM Transformers (CPU) cargado exitosamente!")
        except Exception as e:
            print(f"❌ Error al cargar Transformers: {e}")
            return None, None
            
    return llm_model, llm_tokenizer

@router.post("/chat")
async def chat_generation(
    prompt: str = Form(...),
    system_prompt: str = Form("Eres un asistente inteligente y servicial. Responde ÚNICAMENTE en Español o Inglés, de forma clara y directa. No uses otros idiomas."),
    max_tokens: int = Form(1024),
    temperature: float = Form(0.6),
    thread_id: int = Form(None),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    model, tokenizer = get_llm()
    
    target_thread_id = thread_id
    if not target_thread_id:
        new_thread = Thread(title=prompt[:30], user_id=user.id)
        db.add(new_thread)
        db.commit()
        db.refresh(new_thread)
        target_thread_id = new_thread.id
    
    user_msg = Message(thread_id=target_thread_id, role="user", content=prompt)
    db.add(user_msg)
    db.commit()

    history = db.query(Message).filter(Message.thread_id == target_thread_id).order_by(Message.created_at).all()
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h.role, "content": h.content})
    
    async def stream_generator():
        full_response = ""
        current_buffer = ""
        word_count = 0
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if USE_MLX:
            sampler = mlx_lm.sample_utils.make_sampler(temp=temperature)
            token_iterator = (resp.text for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt_formatted, max_tokens=max_tokens, sampler=sampler))
        else:
            inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=None)
            generation_kwargs = dict(
                inputs, 
                streamer=streamer, 
                max_new_tokens=max_tokens, 
                do_sample=True, 
                temperature=temperature
            )
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            token_iterator = streamer

        try:
            for token_text in token_iterator:
                full_response += token_text
                current_buffer += token_text
                yield token_text
                
                if " " in token_text:
                    word_count += 1
                
                is_strong_punctuation = bool(re.search(r'[.!?;:]\s*$', token_text))
                is_comma = bool(re.search(r',\s*$', token_text))
                is_newline = "\n" in token_text
                
                should_flush = False
                if user.default_voice_id:
                    if is_strong_punctuation or is_newline:
                        should_flush = True
                    elif word_count >= 50:
                        should_flush = True
                    elif word_count >= 8 and is_comma:
                        should_flush = True
                        
                    if should_flush:
                        clean_text = current_buffer.strip()
                        if len(clean_text) > 2: 
                            try:
                                audio_io = await get_edge_audio_stream(clean_text, "es-ES-AlvaroNeural")
                                b64_audio = base64.b64encode(audio_io.getvalue()).decode('utf-8')
                                data_url = f"data:audio/mp3;base64,{b64_audio}"
                                yield f"||VOICE_CHUNK:{data_url}||"
                            except Exception as e:
                                print(f"❌ Error generando audio: {e}")
                            
                            current_buffer = ""
                            word_count = 0
            
            if user.default_voice_id and current_buffer.strip():
                try:
                    clean_text = current_buffer.strip()
                    audio_io = await get_edge_audio_stream(clean_text, "es-ES-AlvaroNeural")
                    b64_audio = base64.b64encode(audio_io.getvalue()).decode('utf-8')
                    data_url = f"data:audio/mp3;base64,{b64_audio}"
                    yield f"||VOICE_CHUNK:{data_url}||"
                except:
                    pass
            
            assistant_msg = Message(thread_id=target_thread_id, role="assistant", content=full_response)
            db.add(assistant_msg)
            db.commit()
                
        except Exception as e:
            print(f"Generación Error: {e}")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(stream_generator(), media_type="text/plain")

@router.post("/chat/voice")
async def chat_voice_to_voice(
    audio: UploadFile = File(...),
    system_prompt: str = Form("Eres un asistente inteligente y servicial. Responde ÚNICAMENTE en Español o Inglés de forma clara. Si no entiendes el audio o hay ruido, responde pidiendo aclaración en español."),
    max_tokens: int = Form(1024),
    temperature: float = Form(0.6),
    thread_id: int = Form(None),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    stt_model = get_whisper()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        print("🎙️ Transcribiendo audio (Forzando Español)...")
        result = stt_model.transcribe(tmp_path, language="es", task="transcribe")
        user_text = result["text"].strip()
        print(f"👤 Usuario dijo: {user_text}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not user_text:
        raise HTTPException(status_code=400, detail="No se pudo entender el audio")

    target_thread_id = thread_id
    if not target_thread_id:
        new_thread = Thread(title=user_text[:30], user_id=user.id)
        db.add(new_thread)
        db.commit()
        db.refresh(new_thread)
        target_thread_id = new_thread.id

    user_msg = Message(thread_id=target_thread_id, role="user", content=user_text)
    db.add(user_msg)
    db.commit()

    history = db.query(Message).filter(Message.thread_id == target_thread_id).order_by(Message.created_at).all()
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h.role, "content": h.content})

    async def voice_response_generator():
        model, tokenizer = get_llm()
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        full_response = ""
        current_buffer = ""
        word_count = 0
        
        yield f"||USER_TRANSCRIPTION:{user_text}||\n"

        if USE_MLX:
            sampler = mlx_lm.sample_utils.make_sampler(temp=temperature)
            token_iterator = (resp.text for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt_formatted, max_tokens=max_tokens, sampler=sampler))
        else:
            inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=None)
            generation_kwargs = dict(
                inputs, 
                streamer=streamer, 
                max_new_tokens=max_tokens, 
                do_sample=True, 
                temperature=temperature
            )
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            token_iterator = streamer

        for token_text in token_iterator:
            full_response += token_text
            current_buffer += token_text
            yield token_text
            
            if " " in token_text:
                word_count += 1
            
            if user.default_voice_id:
                is_strong_punctuation = bool(re.search(r'[.!?;:]\s*$', token_text))
                is_comma = bool(re.search(r',\s*$', token_text))
                is_newline = "\n" in token_text
                
                should_flush = False
                if is_strong_punctuation or is_newline:
                    should_flush = True
                elif word_count >= 50:
                    should_flush = True
                elif word_count >= 8 and is_comma:
                    should_flush = True
                        
                if should_flush:
                    clean_text = current_buffer.strip()
                    if len(clean_text) > 2:
                        try:
                            audio_io = await get_edge_audio_stream(clean_text, "es-ES-AlvaroNeural")
                            b64_audio = base64.b64encode(audio_io.getvalue()).decode('utf-8')
                            data_url = f"data:audio/mp3;base64,{b64_audio}"
                            yield f"||VOICE_CHUNK:{data_url}||"
                        except Exception as e:
                             print(f"❌ Error generando audio: {e}")

                        current_buffer = ""
                        word_count = 0

        if user.default_voice_id and current_buffer.strip():
            try:
                clean_text = current_buffer.strip()
                audio_io = await get_edge_audio_stream(clean_text, "es-ES-AlvaroNeural")
                b64_audio = base64.b64encode(audio_io.getvalue()).decode('utf-8')
                data_url = f"data:audio/mp3;base64,{b64_audio}"
                yield f"||VOICE_CHUNK:{data_url}||"
            except:
                pass

        assistant_msg = Message(thread_id=target_thread_id, role="assistant", content=full_response)
        db.add(assistant_msg)
        db.commit()

    return StreamingResponse(voice_response_generator(), media_type="text/plain")

# --- CONVERSATION CRUD ---

@router.post("/threads/")
def create_thread(title: str = Form(...), db: Session = Depends(get_db), user = Depends(get_current_user)):
    thread = Thread(title=title, user_id=user.id)
    db.add(thread)
    db.commit()
    return {"id": thread.id, "title": thread.title}

@router.get("/threads/user/{user_id}")
def get_user_threads(user_id: int, db: Session = Depends(get_db), user = Depends(get_current_user)):
    if user.id != user_id and user.id != 1:
        raise HTTPException(status_code=403, detail="No autorizado")
    return db.query(Thread).filter(Thread.user_id == user_id).all()

@router.patch("/threads/{thread_id}")
def update_thread(thread_id: int, title: str = Form(...), db: Session = Depends(get_db), user = Depends(get_current_user)):
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Hilo no encontrado")
    if thread.user_id != user.id and user.id != 1:
        raise HTTPException(status_code=403, detail="No autorizado")
    thread.title = title
    db.commit()
    return {"status": "actualizado", "title": thread.title}

@router.delete("/threads/{thread_id}")
def delete_thread(thread_id: int, db: Session = Depends(get_db), user = Depends(get_current_user)):
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Hilo no encontrado")
    if thread.user_id != user.id and user.id != 1:
        raise HTTPException(status_code=403, detail="No autorizado")
    db.delete(thread)
    db.commit()
    return {"status": "eliminado"}

@router.post("/messages/")
def create_message(thread_id: int = Form(...), role: str = Form(...), content: str = Form(...), audio_id: str = Form(None), db: Session = Depends(get_db), user = Depends(get_current_user)):
    msg = Message(thread_id=thread_id, role=role, content=content, audio_id=audio_id)
    db.add(msg)
    db.commit()
    return {"id": msg.id}

@router.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: int, db: Session = Depends(get_db), user = Depends(get_current_user)):
    return db.query(Message).filter(Message.thread_id == thread_id).order_by(Message.created_at).all()
