# robust-whisper
An improved version of OpenAI's ASR model whisper. This improved version is 25% secure than the vanilla whisper against signal processing attacks.

## Install
```
pip install git+https://github.com/TeamSODA18/robust-whisper.git
```


## Usage
```
import torchaudio
from robust_whisper.models import RobustWhisper

robust_asr = RobustWhisper()

data, sample_rate = torchaudio.load(r'path\to\audio\file')

print(robust_asr(data, sample_rate))
```

## Leaderboard

<!-- | Dataset     | Attacked Model | Improvement |    Bening   |     MAE     |
|-------------|----------------|-------------|-------------|-------------|
| LibriSpeech |    Whisper     | 22.6467159% | - 2.5054850 |  1246.8671  |
| CommonVoice |    Whisper     | 18.3795567% | -19.5676749 |   643.8285  |
| LibriSpeech |   AssemblyAI   | 10.0086279% | - 5.3726637 |  1907.2000  |
| CommonVoice |   AssemblyAI   |- 1.0471190% | - 4.6581197 |   719.6072  | -->


| Dataset     | Attacked Model | attacked UD |  attacked D |  benign D   |
|-------------|----------------|-------------|-------------|-------------|
| LibriSpeech |    Whisper     |   0.10418   |   0.07295   |   0.01670   |
| CommonVoice |    Whisper     |   0.43696   |   0.36299   |   0.14525   |
| LibriSpeech |   AssemblyAI   |   0.35352   |   0.30016   |   0.02728   |
| CommonVoice |   AssemblyAI   |   0.82819   |   0.59374   |   0.07829   |

## Comparison Demucs
| Dataset     | Attacked Model | attacked UD |  attacked D |  benign D   |
|-------------|----------------|-------------|-------------|-------------|
| LibriSpeech |    Whisper     |   0.10418   |  0.109869   |   0.025169  |
| CommonVoice |    Whisper     |   0.43696   |  0.412723   |   0.158430  |
| LibriSpeech |   AssemblyAI   |   0.35352   |  0.290059   |   -------   |
| CommonVoice |   AssemblyAI   |   0.82819   |  1.443111   |   -------   |

## Comparison MetricGan
| Dataset     | Attacked Model | attacked UD |  attacked D |  benign D   |
|-------------|----------------|-------------|-------------|-------------|
| LibriSpeech |    Whisper     |   0.10418   |  0.15395    |   0.05730   |
| CommonVoice |    Whisper     |   0.43696   |  0.54714    |   0.19573   |
| LibriSpeech |   AssemblyAI   |   0.35352   |  0.41053    |   0.05155   |
| CommonVoice |   AssemblyAI   |   0.82819   | 1.1617340   |   0.24594   |
