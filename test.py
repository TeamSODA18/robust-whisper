import whisper
from datasets import load_dataset
import torch
import json
import numpy as np
from utils import *

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

model = whisper.load_model("base")
encoder = model.encoder
encoder.load_state_dict(torch.load("best_encoder_3.h5"))
encoder.eval()
model.encoder = encoder

original_model = whisper.load_model("base")
original_model = original_model.to(device)

wer_attack_list = []
wer_recon_list = []

i = 0
for attacked, original in test_loader:
    attacked_transcript_new = get_transcript(attacked, model)
    attacked_transcript_original = get_transcript(attacked, original_model)

    original_transcript_new = get_transcript(original, model)
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
        break

wer_attack_arr = np.array(wer_attack_list)
wer_recon_arr = np.array(wer_recon_list)

improvement = wer_attack_arr - wer_recon_arr
wer_attack_arr[wer_attack_arr == 0] = 1
improvement = improvement / wer_attack_arr

print("improvement", np.mean(improvement) * 100, "%")
