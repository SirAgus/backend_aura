
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from database import init_db
from routers import users, chat, voice, openai
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Voice Backend API", version="2.0")

# Firewall / Trusted Host
from fastapi.middleware.trustedhost import TrustedHostMiddleware

allowed_hosts_env = os.getenv("ALLOWED_HOSTS", "*")
allowed_hosts_list = [h.strip() for h in allowed_hosts_env.split(",")]

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=allowed_hosts_list
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Events
@app.on_event("startup")
def on_startup():
    print("🚀 Starting Voice Backend...")
    try:
        init_db()
        print("✅ Database initialized (Tables created)")
        
        # Create default admin if not exists
        from database import SessionLocal, User
        from dependencies import get_password_hash
        from sqlalchemy import text
        
        db = SessionLocal()
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            print("👤 Creating default admin user...")
            hashed_pwd = get_password_hash("upfint2001")
            admin_user = User(id=1, username="admin", email="admin@aura.ai", hashed_password=hashed_pwd, role="admin")
            db.add(admin_user)
            db.commit()
            # PostgreSQL sequence sync: ensures the next registered user doesn't conflict with ID 1
            try:
                db.execute(text("SELECT setval('users_id_seq', (SELECT MAX(id) FROM users))"))
                db.commit()
            except:
                pass # Non-postgres DBs might not need/support this
            print("✅ Default admin created (admin / upfint2001)")
        db.close()
        
        # --- Voice Sync ---
        from database import Voice
        import json
        sync_db = SessionLocal()
        metadata_path = "voices/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                voices_data = json.load(f)
                for v_id, info in voices_data.items():
                    existing = sync_db.query(Voice).filter(Voice.id == v_id).first()
                    if not existing:
                        print(f"🎙️ Syncing voice: {v_id}")
                        new_voice = Voice(
                            id=v_id,
                            owner_id=None,
                            filename=info.get("filename"),
                            language=info.get("language", "en"),
                            region=info.get("region"),
                            gender=info.get("gender"),
                            description=info.get("description")
                        )
                        sync_db.add(new_voice)
                sync_db.commit()
        sync_db.close()
        # ------------------
        
    except Exception as e:
        print(f"❌ Database init failed: {e}")
        
    # Trigger default downloads if needed (moved logic to routers or services if preferable, 
    # but for now we keep it simple or let the router handle it on first request)

# Include Routers
app.include_router(users.router, tags=["Users & Auth"])
app.include_router(chat.router, tags=["Chat & LLM"])
app.include_router(voice.router, tags=["Voice & TTS"])
app.include_router(openai.router, tags=["OpenAI Compatible API"])

# Static for accessing outputs directly if needed
# app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
