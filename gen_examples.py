from turtle import distance
import whisper
from datasets import load_dataset
import torch
from sklearn.manifold import TSNE
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from robust_whisper.models import RobustWhisper
from matplotlib.lines import Line2D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset(
    "TeamSODA/ae-signal_processing_attacks_whisper_commonvoice", split="train"
)

loader = GetXYEval(train_data, device=device)

model = RobustWhisper()
original_model = whisper.load_model("base")
original_model = original_model.to(device)


attacked_list = []
original_list= []
recon_list = []

for attacked, original in loader:
    attacked_transcript_new = model(attacked.squeeze(), 16000)
    attacked_transcript_original = get_transcript(attacked, original_model)
    original_transcript_original = get_transcript(original, original_model)

    wer_attack, wer_recon = get_wer(
        original_transcript_original,
        attacked_transcript_original,
        attacked_transcript_new,
    )

    if wer_attack!=0 and wer_recon<wer_attack:
        print(original_transcript_original)
        print(attacked_transcript_original)
        print(attacked_transcript_new)