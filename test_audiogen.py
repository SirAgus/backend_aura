import sys
import torch
from unittest.mock import MagicMock

# Mock xformers before importing audiocraft
mock_xformers = MagicMock()
mock_xformers_ops = MagicMock()
mock_xformers_ops.unbind = torch.unbind
mock_xformers.ops = mock_xformers_ops
# Add more common ops if needed
sys.modules["xformers"] = mock_xformers
sys.modules["xformers.ops"] = mock_xformers_ops
sys.modules["xformers.checkpoint_fairinternal"] = MagicMock()

try:
    from audiocraft.models import AudioGen
    import torch
    
    print("Loading AudioGen...")
    model = AudioGen.get_pretrained('facebook/audiogen-medium')
    print("AudioGen loaded!")
    
    # Simple generation test
    descriptions = ['birds singing in a forest']
    wav = model.generate(descriptions, progress=True)
    print(f"Generated wav shape: {wav.shape}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
