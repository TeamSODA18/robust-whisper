from datasets import load_dataset
import torch
from utils import *
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset(
    "TeamSODA/ae-signal_processing_attacks_whisper_librispeech", split="train"
)
loader = GetXYEval(train_data, device=device)

criterian = torch.nn.L1Loss(reduction = 'sum')
mae = []
for attacked, original in tqdm(loader):
    if attacked.shape[-1]==original.shape[-1]:
        loss = criterian(attacked, original)
        mae.append(loss)

print("mae", sum(mae)/len(mae))