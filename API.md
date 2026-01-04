# API Reference - Voice Cloning Backend

Documentación completa de todos los endpoints de la API de síntesis y clonación de voz.

---

## 🔐 Autenticación

La API utiliza **HTTP Basic Authentication** para la mayoría de los endpoints.

### Credenciales

Las credenciales se configuran en el archivo `.env`:

```env
API_USERNAME=admin
API_PASSWORD=upfint2001
```

### Cómo autenticarse

#### Con curl:
```bash
curl -u username:password http://localhost:8000/endpoint
```

#### Con Python (requests):
```python
import requests

response = requests.get(
    "http://localhost:8000/endpoint",
    auth=("username", "password")
)
```

#### Con JavaScript (fetch):
```javascript
const response = await fetch('http://localhost:8000/endpoint', {
  headers: {
    'Authorization': 'Basic ' + btoa('username:password')
  }
});
```

---

## 🌍 Soporte de Español

### Modelo TTS Disponible

Solo está disponible **ChatterboxTurboTTS** en la versión actual del paquete:

- **Funcionamiento**: Idioma determinado por voz de referencia
- **Español**: ✅ Funciona con voces chilenas (`agus`, `agus_latin`)
- **Parámetros**: `language` solo selecciona voz apropiada
- **Ventaja**: Siempre disponible y funcional

### Nota sobre Soporte Multilingüe

El modelo multilingüe (`ChatterboxMultilingualTTS`) **no está disponible** en `chatterbox-tts 0.1.6`. Solo está incluido el modelo Turbo, que funciona correctamente con voces de referencia para español.

### Mejorar Acento Español

Para mejor acento, sube voces más naturales:

```bash
# Subir voz española más natural
curl -X POST "http://localhost:8000/voices/upload" \
  -u admin:upfint2001 \
  -F "name=voz_madrid" \
  -F "language=es" \
  -F "region=ES" \
  -F "file=@voz_espanol.wav"
```

### Consejos para español natural:

1. **Texto limpio**: Usa tildes, puntuación correcta y evita mezclas de idiomas
2. **Voces apropiadas**: Elige voces chilenas, argentinas o españolas según el acento deseado
3. **Parámetro language**: Útil especialmente con el modelo multilingüe para forzar pronunciación

---

## 📡 Endpoints

### 1. Login

**Endpoint:** `POST /login`  
**Autenticación:** ❌ No requerida (usa HTTP Basic Auth en la petición)  
**Descripción:** Verifica las credenciales de usuario

#### Request

```bash
curl -u admin:upfint2001 http://localhost:8000/login
```

#### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/login",
    auth=("admin", "upfint2001")
)

result = response.json()
if result["success"]:
    print("Login exitoso!")
else:
    print("Credenciales inválidas")
```

#### Response - Login Exitoso

**Content-Type:** `application/json`  
**Status Code:** `200 OK`

```json
{
  "success": true,
  "message": "Login successful",
  "user": "admin"
}
```

#### Response - Login Fallido

```json
{
  "success": false,
  "message": "Invalid credentials"
}
```

---

### 2. Health Check

**Endpoint:** `GET /`  
**Autenticación:** ❌ No requerida  
**Descripción:** Verifica el estado del servidor

#### Request
```bash
curl http://localhost:8000/
```

#### Response
```json
{
  "status": "Voice Backend Running",
  "model": "ResembleAI/chatterbox-turbo or chatterbox-multilingual",
  "auth": "enabled",
  "tts_model": "turbo or multilingual"
}
```

**Status Code:** `200 OK`

---

### 3. Demo TTS (Sin autenticación)

**Endpoint:** `GET /demo` o `POST /demo`  
**Autenticación:** ❌ No requerida  
**Descripción:** Genera audio temporal sin guardar archivos. Ideal para pruebas rápidas.

#### Request Parameters

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `text` | Query/Form | ✅ Sí | Texto a sintetizar (soporte nativo para español) |
| `language` | Query/Form | ❌ No | Idioma para selección de voz y pronunciación (es/en). Con modelo multilingüe, afecta la pronunciación directamente |
| `voice_id` | Query | ❌ No | ID específico de voz a usar |

#### GET Request Example

```bash
curl "http://localhost:8000/demo?text=Hola%20mundo&language=es" -o demo.wav
```

#### POST Request Example

```bash
curl -X POST "http://localhost:8000/demo" \
  -F "text=Hola mundo" \
  -F "language=es" \
  --output demo.wav
```

#### Response

**Content-Type:** `audio/wav`  
**Status Code:** `200 OK`

Retorna directamente el audio WAV **sin guardarlo en disco**.

#### Notas Importantes

- **No se guarda** el audio generado ni el historial
- Es **temporal** - solo para preview inmediata
- Para guardar audio permanentemente, usa `/generate-tts`

#### Error Responses

| Status Code | Descripción |
|-------------|-------------|
| `503 Service Unavailable` | La voz por defecto no está disponible |
| `500 Internal Server Error` | Error al generar el audio |

---

### 4. Generate TTS

**Endpoint:** `POST /generate-tts`  
**Autenticación:** ✅ Requerida (Basic Auth)  
**Descripción:** Genera audio con una voz personalizada (clonada o guardada)

#### Request Parameters

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `text` | Form Data | ✅ Sí | Texto a sintetizar |
| `audio_prompt` | File (WAV) | ⚠️ Condicional | Archivo de audio para clonar la voz (3-10 segundos recomendado) |
| `voice_id` | Form Data | ⚠️ Condicional | ID de una voz guardada previamente |

**Nota:** Debes proporcionar **`audio_prompt`** O **`voice_id`**, no ambos.

#### Request Example (con voz guardada)

```bash
curl -X POST "http://localhost:8000/generate-tts" \
  -u admin:upfint2001 \
  -F "text=This is a test with a saved voice" \
  -F "voice_id=female_english" \
  --output output.wav
```

#### Request Example (con archivo de audio)

```bash
curl -X POST "http://localhost:8000/generate-tts" \
  -u admin:upfint2001 \
  -F "text=This is a test with voice cloning" \
  -F "audio_prompt=@my_voice_sample.wav" \
  --output output.wav
```

#### Python Example

```python
import requests

# Con voz guardada
response = requests.post(
    "http://localhost:8000/generate-tts",
    auth=("admin", "upfint2001"),
    data={
        "text": "Hello world",
        "voice_id": "female_english"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# Con archivo de audio
with open("voice_sample.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/generate-tts",
        auth=("admin", "upfint2001"),
        data={"text": "Hello world"},
        files={"audio_prompt": audio_file}
    )
```

#### Response

**Content-Type:** `audio/wav`  
**Status Code:** `200 OK`

Retorna directamente el archivo de audio WAV generado.

#### Error Responses

| Status Code | Descripción |
|-------------|-------------|
| `401 Unauthorized` | Credenciales inválidas |
| `404 Not Found` | El `voice_id` especificado no existe |
| `400 Bad Request` | Falta `audio_prompt` o `voice_id` |
| `500 Internal Server Error` | Error al generar el audio |

---

### 5. Upload Voice Clone

**Endpoint:** `POST /voices/upload`  
**Autenticación:** ✅ Requerida (Basic Auth)  
**Descripción:** Guarda una muestra de voz para reutilizarla posteriormente

#### Request Parameters

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `name` | Form Data | ✅ Sí | Nombre identificador de la voz (alfanumérico, guiones y espacios) |
| `file` | File (WAV) | ✅ Sí | Archivo de audio WAV (3-10 segundos recomendado) |

#### Request Example

```bash
curl -X POST "http://localhost:8000/voices/upload" \
  -u admin:upfint2001 \
  -F "name=my_custom_voice" \
  -F "file=@voice_sample.wav"
```

#### Python Example

```python
import requests

with open("voice_sample.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/voices/upload",
        auth=("admin", "upfint2001"),
        data={"name": "my_custom_voice"},
        files={"file": audio_file}
    )

print(response.json())
```

#### Response

**Content-Type:** `application/json`  
**Status Code:** `200 OK`

```json
{
  "status": "Voice clone saved locally",
  "voice_id": "my_custom_voice"
}
```

#### Error Responses

| Status Code | Descripción |
|-------------|-------------|
| `401 Unauthorized` | Credenciales inválidas |
| `400 Bad Request` | Nombre de voz inválido |

---

### 6. List Voices

**Endpoint:** `GET /voices`  
**Autenticación:** ✅ Requerida (Basic Auth)  
**Descripción:** Lista todas las voces guardadas en el sistema

#### Request Example

```bash
curl -X GET "http://localhost:8000/voices" \
  -u admin:upfint2001
```

#### Python Example

```python
import requests

response = requests.get(
    "http://localhost:8000/voices",
    auth=("admin", "upfint2001")
)

voices = response.json()
print(voices)
```

#### Response

**Content-Type:** `application/json`  
**Status Code:** `200 OK`

```json
{
  "voices": [
    "female_english",
    "male_english",
    "my_custom_voice"
  ]
}
```

#### Error Responses

| Status Code | Descripción |
|-------------|-------------|
| `401 Unauthorized` | Credenciales inválidas |

---

### 8. Get History

**Endpoint:** `GET /history`  
**Autenticación:** ✅ Requerida (Basic Auth)  
**Descripción:** Obtiene el historial completo de generaciones de audio

#### Request Example

```bash
curl -X GET "http://localhost:8000/history" \
  -u admin:upfint2001
```

#### Python Example

```python
import requests

response = requests.get(
    "http://localhost:8000/history",
    auth=("admin", "upfint2001")
)

history = response.json()
for entry in history:
    print(f"{entry['timestamp']}: {entry['text'][:50]}...")
```

#### Response

**Content-Type:** `application/json`  
**Status Code:** `200 OK`

```json
[
  {
    "id": "abc123-def456",
    "text": "Hello, this is a test",
    "filename": "gen_abc123-def456.wav",
    "voice_used": "female_english",
    "timestamp": "2026-01-04T13:30:00.123456",
    "user": "admin"
  },
  {
    "id": "xyz789-uvw012",
    "text": "Another test message",
    "filename": "demo_xyz789-uvw012.wav",
    "voice_used": "female_english",
    "timestamp": "2026-01-04T13:45:00.654321",
    "user": "anonymous"
  }
]
```

#### Response Fields

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | string | UUID único de la generación |
| `text` | string | Texto que fue sintetizado |
| `filename` | string | Nombre del archivo de audio generado |
| `voice_used` | string | ID de la voz utilizada o "uploaded_sample" |
| `timestamp` | string | Fecha y hora ISO 8601 |
| `user` | string | Usuario que realizó la generación ("anonymous" para `/demo`) |

#### Error Responses

| Status Code | Descripción |
|-------------|-------------|
| `401 Unauthorized` | Credenciales inválidas |

---

### 9. List Available Voices

**Endpoint:** `GET /voices/list`  
**Autenticación:** ❌ No requerida  
**Descripción:** Lista todas las voces disponibles con sus metadatos. Usa el endpoint `/demo` para probarlas.

#### Request Example

```bash
curl http://localhost:8000/voices/list
```

#### Python Example

```python
import requests

response = requests.get("http://localhost:8000/voices/list")
voices = response.json()

for voice in voices["voices"]:
    print(f"Voice: {voice['id']}")
    print(f"Language: {voice['language']}")
    print(f"Region: {voice['region']}")
    print(f"Preview URL: http://localhost:8000{voice['preview_url']}")
    print("---")
```

#### Response

**Content-Type:** `application/json`  
**Status Code:** `200 OK`

```json
{
  "voices": [
    {
      "id": "agus",
      "name": "agus",
      "filename": "agus.wav",
      "language": "es",
      "region": "CL",
      "gender": "male",
      "description": "Spanish Male Voice (Chile)",
      "preview_url": "/demo?text=Hola%20mundo&voice_id=agus"
    }
  ],
  "total": 4
}
```

### 10. Download Audio File

**Endpoint:** `GET /download/{filename}`  
**Autenticación:** ✅ Requerida (Basic Auth)  
**Descripción:** Descarga un archivo de audio específico del historial

#### Request Parameters

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `filename` | Path | ✅ Sí | Nombre del archivo (obtenido del historial) |

#### Request Example

```bash
curl -X GET "http://localhost:8000/download/gen_abc123-def456.wav" \
  -u admin:upfint2001 \
  --output downloaded_audio.wav
```

#### Python Example

```python
import requests

# Primero obtener el historial
history_response = requests.get(
    "http://localhost:8000/history",
    auth=("admin", "upfint2001")
)
history = history_response.json()

# Descargar el primer archivo
if history:
    filename = history[0]["filename"]
    audio_response = requests.get(
        f"http://localhost:8000/download/{filename}",
        auth=("admin", "upfint2001")
    )
    
    with open(f"downloaded_{filename}", "wb") as f:
        f.write(audio_response.content)
```

#### Response

**Content-Type:** `audio/wav`  
**Status Code:** `200 OK`

Retorna directamente el archivo de audio WAV.

#### Error Responses

| Status Code | Descripción |
|-------------|-------------|
| `401 Unauthorized` | Credenciales inválidas |
| `404 Not Found` | El archivo no existe |

---

## 📊 Códigos de Estado HTTP

| Código | Significado | Cuándo ocurre |
|--------|-------------|---------------|
| `200 OK` | Éxito | La solicitud se procesó correctamente |
| `400 Bad Request` | Solicitud inválida | Faltan parámetros requeridos o son inválidos |
| `401 Unauthorized` | No autorizado | Credenciales incorrectas o faltantes |
| `404 Not Found` | No encontrado | El recurso solicitado no existe |
| `500 Internal Server Error` | Error del servidor | Error al procesar la solicitud (ej: modelo no cargado) |
| `503 Service Unavailable` | Servicio no disponible | Recurso temporal no disponible |

---

## 🔄 Tipos de Contenido

### Request Content-Types

- **Form Data:** `multipart/form-data` (para endpoints con archivos)
- **JSON:** `application/json` (no usado actualmente)

### Response Content-Types

- **Audio:** `audio/wav` (archivos de audio)
- **JSON:** `application/json` (respuestas de datos)

---

## 🚀 Ejemplos de Integración

### JavaScript (Frontend)

```javascript
// Demo sin autenticación
async function generateDemo(text) {
  const formData = new FormData();
  formData.append('text', text);
  
  const response = await fetch('http://localhost:8000/demo', {
    method: 'POST',
    body: formData
  });
  
  const audioBlob = await response.blob();
  const audioUrl = URL.createObjectURL(audioBlob);
  
  const audio = new Audio(audioUrl);
  audio.play();
}

// Con autenticación
async function generateTTS(text, voiceId, username, password) {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('voice_id', voiceId);
  
  const response = await fetch('http://localhost:8000/generate-tts', {
    method: 'POST',
    headers: {
      'Authorization': 'Basic ' + btoa(`${username}:${password}`)
    },
    body: formData
  });
  
  const audioBlob = await response.blob();
  return URL.createObjectURL(audioBlob);
}
```

### Python (Script completo)

```python
import requests
from pathlib import Path

class VoiceAPI:
    def __init__(self, base_url="http://localhost:8000", username="admin", password="upfint2001"):
        self.base_url = base_url
        self.auth = (username, password)
    
    def demo(self, text, output_file="demo.wav"):
        """Genera audio sin autenticación"""
        response = requests.post(
            f"{self.base_url}/demo",
            data={"text": text}
        )
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        return output_file
    
    def generate(self, text, voice_id=None, audio_prompt_path=None, output_file="output.wav"):
        """Genera audio con voz personalizada"""
        data = {"text": text}
        files = {}
        
        if voice_id:
            data["voice_id"] = voice_id
        elif audio_prompt_path:
            files["audio_prompt"] = open(audio_prompt_path, "rb")
        
        response = requests.post(
            f"{self.base_url}/generate-tts",
            auth=self.auth,
            data=data,
            files=files
        )
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        return output_file
    
    def upload_voice(self, name, audio_file_path):
        """Sube una nueva voz"""
        with open(audio_file_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/voices/upload",
                auth=self.auth,
                data={"name": name},
                files={"file": f}
            )
        response.raise_for_status()
        return response.json()
    
    def list_voices(self):
        """Lista todas las voces disponibles"""
        response = requests.get(
            f"{self.base_url}/voices",
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()["voices"]
    
    def get_history(self):
        """Obtiene el historial"""
        response = requests.get(
            f"{self.base_url}/history",
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()

# Uso
api = VoiceAPI()

# Demo rápido
api.demo("Hello world", "demo.wav")

# Generar con voz guardada
api.generate("This is a test", voice_id="female_english", output_file="test.wav")

# Subir nueva voz
api.upload_voice("my_voice", "sample.wav")

# Listar voces
voices = api.list_voices()
print(f"Available voices: {voices}")
```

---

## 🔒 Seguridad

### Recomendaciones

1. **Cambiar credenciales por defecto** antes de usar en producción
2. **Usar HTTPS** en producción (configurar reverse proxy con nginx/caddy)
3. **Limitar acceso** por IP si es posible
4. **Rotar credenciales** periódicamente
5. **No exponer** el archivo `.env` en repositorios públicos

### CORS

El servidor permite todas las origenes (`*`) por defecto. Para producción, modifica `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-dominio.com"],  # Cambiar aquí
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📝 Notas Adicionales

- **Formato de audio:** Todos los archivos son WAV (sin comprimir)
- **Sample rate:** Depende del modelo (generalmente 24kHz)
- **Límite de texto:** No hay límite estricto, pero textos muy largos pueden tardar más
- **Calidad de clonación:** Mejores resultados con audios limpios de 5-10 segundos
- **Almacenamiento:** Los archivos se guardan indefinidamente hasta que los elimines manualmente
