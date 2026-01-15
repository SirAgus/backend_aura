
import requests
import os
from pathlib import Path

# URLs de alta calidad
SOURCES = {
    "rain": "https://www.soundjay.com/nature/sounds/rain-03.mp3",
    "birds": "https://www.soundjay.com/nature/sounds/canary-singing-01.mp3", 
    "storm": "https://www.soundjay.com/nature/sounds/thunder-01.mp3",
    "office": "https://www.soundjay.com/misc/sounds/busy-office-ambience-1.mp3",
    # Alternative URL for forest since SoundJay might block sometimes
    "forest": "https://cdn.pixabay.com/download/audio/2021/09/06/audio_3606f7dfba.mp3",
    "wind": "https://www.soundjay.com/nature/sounds/wind-howl-01.mp3",
    "cafe": "https://www.soundjay.com/misc/sounds/restaurant-1.mp3",
    "lofi": "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3"
}

AMBIENCE_DIR = "ambience"
os.makedirs(AMBIENCE_DIR, exist_ok=True)
headers = {"User-Agent": "Mozilla/5.0"}

print("⬇️  Forzando descarga de todos los ambientes...")

for name, url in SOURCES.items():
    print(f"🎵  Procesando: {name}...")
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            # Save as temp mp3
            temp_path = Path(AMBIENCE_DIR) / f"temp_{name}.mp3"
            final_path = Path(AMBIENCE_DIR) / f"{name}.wav"
            
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            # Convert to wav using ffmpeg directly to be sure
            import subprocess
            cmd = ["ffmpeg", "-y", "-i", str(temp_path), "-ar", "24000", "-ac", "1", str(final_path)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            print(f"   ✅ {name}: OK")
        else:
            print(f"   ❌ {name}: Falló HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ {name}: Error {e}")

print("✨  ¡Listo! Reinicia el servidor para usar los nuevos sonidos.")
