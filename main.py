
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from database import init_db
from routers import users, chat, voice
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
        
        db = SessionLocal()
        admin = db.query(User).filter(User.id == 1).first()
        if not admin:
            print("👤 Creating default admin user...")
            hashed_pwd = get_password_hash("upfint2001")
            admin_user = User(id=1, username="admin", hashed_password=hashed_pwd)
            db.add(admin_user)
            db.commit()
            print("✅ Default admin created (admin / upfint2001)")
        db.close()
        
    except Exception as e:
        print(f"❌ Database init failed: {e}")
        
    # Trigger default downloads if needed (moved logic to routers or services if preferable, 
    # but for now we keep it simple or let the router handle it on first request)

# Include Routers
app.include_router(users.router, tags=["Users & Auth"])
app.include_router(chat.router, tags=["Chat & LLM"])
app.include_router(voice.router, tags=["Voice & TTS"])

# Static for accessing outputs directly if needed
# app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
