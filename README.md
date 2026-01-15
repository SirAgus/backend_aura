# Voice Backend API

Backend robusto para síntesis de voz, clonación de audio e inteligencia artificial conversacional. Diseñado para ser escalable, seguro y fácil de desplegar.

## 1. De qué trata
Esta API integra múltiples tecnologías de IA y gestión de datos:
- **TTS & Voice Cloning**: Usa **Chatterbox-Turbo** para generar voz y clonar voces a partir de referencias (`.wav`).
- **LLM Local**: Integra **Qwen 2.5 3B** para generación de texto inteligente y chat.
- **Gestión de Base de Datos**: PostgreSQL para usuarios, hilos de chat, mensajes y metadatos de audios.
- **Seguridad**: Autenticación vía **JWT** y encriptación de contraseñas con **BCrypt**.
- **Ambientes Dinámicos**: Mezcla inteligente de sonidos de fondo (lluvia, oficina, etc.) usando fuentes de alta calidad.

## 2. Requisitos Previos

### Opción A (Recomendada): Docker
- **Docker** y **Docker Compose** instalados.
- 4GB+ de RAM asignada a Docker (para cargar modelos de IA).

### Opción B: Ejecución Local (Bare Metal)
- **Sistema Operativo**: macOS (Apple Silicon) o Linux.
- **Python**: 3.11 (Estricto).
- **Gestor de Paquetes**: [PDM](https://pdm-project.org/).
- **Base de Datos**: PostgreSQL 15+ corriendo localmente.
- **Herramientas de Sistema**: `ffmpeg` (Requerido para procesamiento de audio. Instalar con `brew install ffmpeg`).

## 3. Configuración (.env)
Crea un archivo `.env` en la raíz con las siguientes variables:

```bash
# Servidor & Seguridad
SECRET_KEY=tu_super_secreto_para_jwt_cambialo

# Base de Datos
# Uso local:
DATABASE_URL=postgresql://user:password@localhost:5432/voice_db
# Uso Docker:
# DATABASE_URL=postgresql://voice_user:voice_password@db:5432/voice_db

# Almacenamiento
STORAGE_DIR=outputs
VOICES_DIR=voices

# Modelos IA
HF_TOKEN=tu_token_de_huggingface # Opcional, para descarga rápida

# Seguridad Avanzada (Opcional)
ALLOWED_HOSTS="*" # Lista separada por comas, ej: "api.midominio.com,localhost"

```

## 4. Iniciar el Proyecto y Despliegue

### 🛠️ Desarrollo Local (Vía PDM)
Ideal para programar y debuggear.
1. Instalar dependencias:
   ```bash
   pdm install
   ```
2. Asegúrate de tener Postgres corriendo y configurar `.env`.
3. Iniciar servidor (Modo recarga automática):
   ```bash
   pdm run dev
   ```

### 🐳 Docker Compose (Recomendado para inicio rápido)
Levanta la BD y el Backend en contenedores aislados.
```bash
docker-compose up --build
```
- La primera vez tomará tiempo (descarga de imagen base).
- La API estará en `http://localhost:8000`.

### 🚀 Despliegue en Producción

#### Opción A: Railway (Fácil)
1. Conecta tu repositorio a Railway.
2. Railway detectará el `Dockerfile` automáticamente.
3. Agrega un servicio de **PostgreSQL** dentro de Railway.
4. En las **Variables** del servicio Backend, configura:
   - `DATABASE_URL`: (Variable interna de Railway hacia Postgres)
   - `SECRET_KEY`: (Tu clave segura)
   - `HF_TOKEN`: (Tu token de HuggingFace)
5. **Listo**: Railway construirá y desplegará el servicio.

#### Opción B: VPS / Servidor Docker (Generic)
Para desplegar en cualquier servidor Linux con Docker:

1. **Construir la Imagen**:
   ```bash
   docker build -t voice-backend .
   ```

2. **Ejecutar Contenedor**:
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
     -e SECRET_KEY="prod_secret" \
     --name voice-service \
     voice-backend
   ```
   *(Asegúrate de que el contenedor tenga acceso a la red de tu Base de Datos).*



## 5. Documentación de API
La documentación detallada de endpoints, payloads y respuestas se encuentra en la carpeta `docs/`.

- [Referencia de API (Endpoints & Payloads)](docs/API_REFERENCE.md)

También puedes ver la documentación interactiva (Swagger UI) al iniciar el servidor en:
- `http://localhost:8000/docs`
