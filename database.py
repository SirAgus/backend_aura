
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database Config
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://voice_user:voice_password@localhost:5432/voice_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# --- MODELS ---

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="usuario")
    default_voice_id = Column(String, ForeignKey("voices.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    voices = relationship("Voice", back_populates="owner", foreign_keys="Voice.owner_id")
    audios = relationship("Audio", back_populates="user")
    threads = relationship("Thread", back_populates="user")

class Voice(Base):
    """
    Voces clonadas.
    Convention: if owner_id is NULL, it is a PUBLIC voice.
    """
    __tablename__ = "voices"
    
    id = Column(String, primary_key=True, index=True) # Name as ID (e.g. 'agus')
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Null = Public
    filename = Column(String)
    language = Column(String, default="en")
    region = Column(String, nullable=True)
    gender = Column(String, nullable=True) # Addition
    description = Column(String, nullable=True) # Addition
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="voices", foreign_keys=[owner_id])

class Audio(Base):
    """
    Historial de audios generados.
    """
    __tablename__ = "audios"
    
    id = Column(String, primary_key=True) # UUID
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    content = Column(Text) # Full text content
    voice_used = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="audios")

class Thread(Base):
    """
    Hilos de conversación (Chat).
    """
    __tablename__ = "threads"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="threads")
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")

class Message(Base):
    """
    Mensajes dentro de un hilo.
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("threads.id"))
    role = Column(String) # 'user' or 'assistant'
    content = Column(Text)
    audio_id = Column(String, ForeignKey("audios.id"), nullable=True) # Optional link to generated audio
    created_at = Column(DateTime, default=datetime.utcnow)
    
    thread = relationship("Thread", back_populates="messages")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
