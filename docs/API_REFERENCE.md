# Referencia de API

Esta documentación detalla los endpoints disponibles, los payloads requeridos y las estructuras de respuesta. Todos los endpoints protegidos requieren el header:
`Authorization: Bearer <access_token>`

---

## 🔐 Autenticación

### Login
Genera un token de acceso JWT.

- **Endpoint**: `POST /auth/token`
- **Content-Type**: `application/x-www-form-urlencoded`
- **Payload**:
  - `username`: string
  - `password`: string
- **Respuesta (200 OK)**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1Ni...",
    "token_type": "bearer",
    "user_id": 1
  }
  ```

### SignUp (Registro)
Crea un nuevo usuario y devuelve el token de sesión inmediatamente.

- **Endpoint**: `POST /auth/signup`
- **Form Data**:
  - `username`: string
  - `password`: string
- **Respuesta (200 OK)**:
  ```json
  {
    "id": 2,
    "username": "nuevo_user",
    "access_token": "...",
    "token_type": "bearer"
  }
  ```

---

## 🗣️ Síntesis de Voz (TTS)

### Generar Audio (TTS)
Genera voz a partir de texto, con opción de clonación y clonado de ambiente.

- **Endpoint**: `POST /generate-tts`
- **Auth**: Requerida.
- **Form Data**:
  - `text`: "Texto a hablar [rain] con ambiente [rain]."
  - `voice_id`: "agus" (o ID de voz guardada)
  - `audio_prompt`: (Archivo .wav opcional si no usas voice_id)
  - `language`: "es" (para modelo multilingual, turbo lo detecta auto)
  - `temperature`: 0.7 (Creatividad)
  - `ambience_id`: "rain", "office", "lofi", etc. (Opcional, fondo constante)
- **Respuesta**: Archivo binario `.wav`.

### Subir Clon de Voz
- **Endpoint**: `POST /voices/upload`
- **Auth**: Requerida.
- **Form Data**:
  - `name`: "nombre_voz"
  - `file`: (Archivo .wav de referencia)
  - `language`: "en" | "es" (Idioma de la voz)
- **Respuesta**:
  ```json
  {
    "status": "saved",
    "voice_id": "nombre_voz_sanitizado"
  }
  ```

---

## 🧠 Inteligencia Artificial (LLM)

### Todo-Chat (Texto)
Genera texto con el modelo local Qwen 2.5 3B.

- **Endpoint**: `POST /chat`
- **Auth**: Requerida.
- **Form Data**:
  - `prompt`: "Escribe un poema sobre la lluvia"
  - `system_prompt`: "Eres un poeta melancólico"
  - `max_tokens`: 200
  - `temperature`: 0.7
- **Respuesta**:
  ```json
  {
    "response": "En el gris lamento de la tarde...",
    "model": "Qwen/Qwen2.5-3B-Instruct"
  }
  ```

---

## 👥 Usuarios

### Obtener Perfil
- **Endpoint**: `GET /users/{user_id}`
- **Respuesta**: `{ "id": 1, "username": "admin", "created_at": "..." }`

### Actualizar Usuario
- **Endpoint**: `PUT /users/{user_id}`
- **Form Data**: `password` (Opcional)

### Eliminar Usuario
- **Endpoint**: `DELETE /users/{user_id}`

---

## 🧵 Hilos de Chat (Threads)

### Crear Hilo
- **Endpoint**: `POST /threads/`
- **Form Data**:
  - `title`: "Conversación sobre voces"
  - `user_id`: 1
- **Respuesta**: `{ "id": 5, "title": "Conversación sobre voces" }`

### Listar mis Hilos
- **Endpoint**: `GET /threads/user/{user_id}`
- **Respuesta**: `[ { "id": 5, "title": "..." }, ... ]`

### Modificar/Borrar Hilo
- `PUT /threads/{id}` (param: `title`)
- `DELETE /threads/{id}`

---

## 💬 Mensajes

### Enviar Mensaje
Guarda un mensaje en un hilo.

- **Endpoint**: `POST /messages/`
- **Form Data**:
  - `thread_id`: 5
  - `role`: "user" | "assistant"
  - `content`: "Hola IA"
  - `audio_id`: (Opcional, si el mensaje tiene un audio generado asociado)
- **Respuesta**: `{ "id": 42 }`

### Leer Mensajes de un Hilo
- **Endpoint**: `GET /threads/{thread_id}/messages`
- **Respuesta**: Lista de objetos mensaje ordenados por fecha.
