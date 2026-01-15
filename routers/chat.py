
from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from database import get_db, Thread, Message
from dependencies import get_current_user
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

router = APIRouter()

# --- LLM SERVICE (Simplified in router for now) ---
llm_model = None
llm_tokenizer = None
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def get_llm():
    global llm_model, llm_tokenizer
    if llm_model is None:
        try:
            print(f"🧠 Loading LLM: {LLM_MODEL_ID}...")
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID, 
                torch_dtype=torch.float16, 
                device_map=device
            )
            print(f"✅ LLM loaded on {device}!")
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            return None, None
    return llm_model, llm_tokenizer

@router.post("/chat")
async def chat_generation(
    prompt: str = Form(...),
    system_prompt: str = Form("You are a helpful assistant."),
    max_tokens: int = Form(200),
    temperature: float = Form(0.7),
    user = Depends(get_current_user)
):
    model, tokenizer = get_llm()
    if not model:
        raise HTTPException(status_code=500, detail="LLM model could not be loaded.")
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9
        )
        generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {"response": response_text, "model": LLM_MODEL_ID}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

# --- CONVERSATION CRUD ---

@router.post("/threads/")
def create_thread(title: str = Form(...), db: Session = Depends(get_db), user = Depends(get_current_user)):
    thread = Thread(title=title, user_id=user.id)
    db.add(thread)
    db.commit()
    return {"id": thread.id, "title": thread.title}

@router.get("/threads/user/{user_id}") # Optional: Could be just /threads/me
def get_user_threads(user_id: int, db: Session = Depends(get_db), user = Depends(get_current_user)):
    # Check auth
    if user.id != user_id and user.id != 1:
        raise HTTPException(status_code=403, detail="Not authorized")
    return db.query(Thread).filter(Thread.user_id == user_id).all()

@router.put("/threads/{thread_id}")
def update_thread(thread_id: int, title: str = Form(...), db: Session = Depends(get_db), user = Depends(get_current_user)):
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.user_id != user.id and user.id != 1:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    thread.title = title
    db.commit()
    return {"status": "updated", "title": thread.title}

@router.delete("/threads/{thread_id}")
def delete_thread(thread_id: int, db: Session = Depends(get_db), user = Depends(get_current_user)):
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    if thread.user_id != user.id and user.id != 1:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    db.delete(thread)
    db.commit()
    return {"status": "deleted"}

@router.post("/messages/")
def create_message(thread_id: int = Form(...), role: str = Form(...), content: str = Form(...), audio_id: str = Form(None), db: Session = Depends(get_db), user = Depends(get_current_user)):
    # Validate ownership?
    msg = Message(thread_id=thread_id, role=role, content=content, audio_id=audio_id)
    db.add(msg)
    db.commit()
    return {"id": msg.id}

@router.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: int, db: Session = Depends(get_db), user = Depends(get_current_user)):
    # Validate ownership?
    return db.query(Message).filter(Message.thread_id == thread_id).order_by(Message.created_at).all()
