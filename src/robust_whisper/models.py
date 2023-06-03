import os
from typing import Any
import requests
import torch
import torchaudio
from torchaudio.transforms import Resample
import whisper


class RobustWhisper:
    def __init__(
        self,
    ):
        if not os.path.isfile("RobustEncoder_2.h5"):
            url = "https://drive.google.com/uc?export=download&confirm=t&id=12mpYFsRPh3xdWAQezx3tScpCXXQFhDA9&uuid=b188fd4a-2a3e-45d7-a547-7a0bc9ad487a&at=AKKF8vyWEQoRU4fWAvjx2-V4ApXh:1685790524942"
            r = requests.get(url, allow_redirects=True)
            open("RobustEncoder_2.h5", "wb").write(r.content)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robust_whisper = whisper.load_model("base")
        self.robust_whisper = self.robust_whisper.to(self.device)
        self.robust_whisper.encoder.load_state_dict(torch.load("RobustEncoder_2.h5", map_location=self.device))

    def get_mel(self, X):
        X = whisper.pad_or_trim(X)
        X = whisper.log_mel_spectrogram(X)
        return X

    def get_transcript(self, X):
        mel = self.get_mel(X)
        # detect the spoken language
        _, probs = self.robust_whisper.detect_language(mel)

        # decode the audio
        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(self.robust_whisper, mel, options)
        # print the recognized text
        return result.text


    def __call__(self, data: torch.Tensor, sample_rate: int):
        data = data.squeeze().to(self.device)
        resample = Resample(sample_rate, 16000)
        data = resample(data)
        transcription = self.get_transcript(data)
        return transcription
