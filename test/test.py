import torchaudio
import robust_whisper
from robust_whisper.models import RobustWhisper

robust_whisper_model = RobustWhisper()

data, sr = torchaudio.load(r"test_audio/attack/33_after_clipping_23.wav")
print(robust_whisper_model(data, sr))