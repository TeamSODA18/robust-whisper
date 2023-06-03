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

| Dataset     | Attacked Model | Improvement |
|-------------|----------------|-------------|
| LibriSpeech |    Whisper     | 22.6467159% |
|-------------|----------------|-------------|
|             |                |             |
|-------------|----------------|-------------|
|             |                |             |

