from spClassify.infer import SpMelCl
from robust_whisper.models import RobustWhisper
import torchaudio
import torch
from datasets import load_dataset
from utils import *

robust_asr = RobustWhisper()
classifier = SpMelCl()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset(
    #"TeamSODA/ae-signal_processing_attacks_whisper_librispeech", split="train"
    #"TeamSODA/ae-signal_processing_attacks_whisper_commonvoice", split="train"
    #"TeamSODA/ae-signal_processing_attacks_assembly_librispeech", split="train"
    "TeamSODA/ae-signal_processing_attacks_assembly_commonvoice", split="train"
)
loader = GetXYEval(train_data, device=device)
#generator = torch.Generator().manual_seed(42)
#_, _, loader = torch.utils.data.random_split(
#    loader,
#    [
#        int(len(loader) * 0.6),
#        int(len(loader) * 0.2),
#        int(len(loader) * 0.2),
#    ],
#    generator,
#)

model = RobustWhisper()

original_model = whisper.load_model("base")
original_model = original_model.to(device)

original_defended = []

for attacked, original in loader:
    original_text = get_transcript(original, original_model, device)

    
    defended_text = model(original.squeeze(), 16000)
    
    wer_recon = get_wer_single(original_text, defended_text)
    original_defended.append(wer_recon)
    print("benign", wer_recon)

print("original defended wer", sum(original_defended)/len(original_defended))