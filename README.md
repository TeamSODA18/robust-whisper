# robust-whisper
An improved version of OpenAI's ASR model whisper. This improved version is 25% secure than the vanilla whisper against signal processing attacks.

## Install
```
pip install git+https://github.com/TeamSODA18/robust-whisper.git
```


## Usage
```
import torchaudio
from robustwhisper.models import RobustWhisper

robust_asr = RobustWhisper()

data, sample_rate = torchaudio.load(r'path\to\audio\file')

print(robust_asr(data, sample_rate))
```

## Leaderboard

| Dataset     | Attacked Model | Improvement |     MAE     |
|-------------|----------------|-------------|-------------|
| LibriSpeech |    Whisper     | 22.6467159% |  1246.8671  |
|-------------|----------------|-------------|-------------|
| CommonVoice |    Whisper     | 18.3795567% |   643.8285  |
|-------------|----------------|-------------|-------------|
| LibriSpeech |   AssemblyAI   | 10.0086279% |  1907.2000  |
|-------------|----------------|-------------|-------------|
| CommonVoice |   AssemblyAI   |- 1.0471190% |   719.6072  |
|-------------|----------------|-------------|-------------|

