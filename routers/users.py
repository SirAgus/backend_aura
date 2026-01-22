
from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from database import get_db, User
from dependencies import get_current_user, create_access_token, create_refresh_token, get_password_hash, verify_password, ACCESS_TOKEN_EXPIRE_MINUTES, generate_api_key
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta, datetime
from pydantic import BaseModel

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str = "usuario"
    api_key: str | None = None
    default_voice_id: str | None = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class SettingsUpdate(BaseModel):
    default_voice_id: str | None = None


router = APIRouter()

# --- AUTH ---

from sqlalchemy import func

@router.post("/auth/token") 
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    login_id_lower = form_data.username.lower()
    # Find by username (case-insensitive) OR email (lowercase)
    user = db.query(User).filter(
        (func.lower(User.username) == login_id_lower) | (User.email == login_id_lower)
    ).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "token_type": "bearer", 
        "user_id": user.id,
        "user": UserOut.model_validate(user)
    }

@router.post("/auth/signup")
def create_user_signup(
    username: str = Form(...), 
    email: str = Form(...), 
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    username_display = username # Preserve casing as written
    email_lower = email.lower() # Email always lowercase
    
    # Uniqueness check (Case-insensitive for username, lowercase for email)
    existing = db.query(User).filter(
        (func.lower(User.username) == username_display.lower()) | 
        (User.email == email_lower)
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="El nombre de usuario o email ya está registrado")
    
    hashed = get_password_hash(password)
    user = User(username=username_display, email=email_lower, hashed_password=hashed)
    db.add(user)
    db.commit()
    
    return {
        "id": user.id, 
        "username": user.username, 
        "email": user.email,
        "role": user.role,
        "status": "usuario creado exitosamente"
    }

# --- USER CRUD ---

@router.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"user": UserOut.model_validate(current_user)}

@router.get("/users/{user_id}", response_model=UserOut)
def read_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user

@router.put("/users/{user_id}")
def update_user(user_id: int, password: str = Form(None), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Security: only allow users to update themselves or admin
    if current_user.id != user_id and current_user.id != 1:
        raise HTTPException(status_code=403, detail="No autorizado para actualizar este usuario")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if password:
        user.hashed_password = get_password_hash(password)
    db.commit()
    return {"status": "actualizado", "user_id": user.id}

@router.patch("/users/me/settings")
def update_settings(settings: SettingsUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    user = db.query(User).filter(User.id == current_user.id).first()
    if settings.default_voice_id is not None:
        user.default_voice_id = settings.default_voice_id
    db.commit()
    return {"status": "configuración actualizada", "default_voice_id": user.default_voice_id}

@router.post("/users/me/api-key")
def create_api_key(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    user = db.query(User).filter(User.id == current_user.id).first()
    user.api_key = generate_api_key()
    db.commit()
    return {"api_key": user.api_key}

@router.get("/users/me/api-key")
def get_api_key(current_user: User = Depends(get_current_user)):
    return {"api_key": current_user.api_key}

@router.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.id != 1: # Only admin can delete users for now
         raise HTTPException(status_code=403, detail="Solo el administrador puede eliminar usuarios")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    db.delete(user)
    db.commit()
    return {"status": "eliminado"}
