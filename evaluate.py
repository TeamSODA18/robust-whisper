import whisper
from datasets import load_dataset
import torch
import numpy as np
from utils import *
from robust_whisper.models import RobustWhisper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset(
    "TeamSODA/ae-signal_processing_attacks_whisper", split="train"
)
train_loader = GetXYEval(train_data, device=device)
generator = torch.Generator().manual_seed(42)
train_loader, val_loader, test_loader = torch.utils.data.random_split(
    train_loader,
    [
        int(len(train_loader) * 0.6),
        int(len(train_loader) * 0.2),
        int(len(train_loader) * 0.2),
    ],
    generator,
)

model = RobustWhisper()

original_model = whisper.load_model("base")
original_model = original_model.to(device)

wer_attack_list = []
wer_recon_list = []

i = 0
for attacked, original in test_loader:
    attacked_transcript_new = model(attacked.squeeze(), 16000)
    attacked_transcript_original = get_transcript(attacked, original_model)

    original_transcript_original = get_transcript(original, original_model)

    wer_attack, wer_recon = get_wer(
        original_transcript_original,
        attacked_transcript_original,
        attacked_transcript_new,
    )
    if wer_attack == 0:
        continue
    wer_attack_list.append(wer_attack)
    wer_recon_list.append(wer_recon)
    print(wer_attack, "->", wer_recon)
    i += 1
    if i == 200:
        pass

wer_attack_arr = np.array(wer_attack_list)
wer_recon_arr = np.array(wer_recon_list)

improvement = wer_attack_arr - wer_recon_arr
wer_attack_arr[wer_attack_arr == 0] = 1
improvement = improvement / wer_attack_arr

print("improvement", np.mean(improvement) * 100, "%")