import whisper
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import random
import numpy as np
from utils import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
train_data = load_dataset("TeamSODA/ae-signal_processing_attacks_whisper", split="train")
train_loader = GetXY(train_data, device=device)
generator = torch.Generator().manual_seed(42)
train_loader, val_loader, test_loader = torch.utils.data.random_split(train_loader, [int(len(train_loader)*0.6), int(len(train_loader)*0.2), int(len(train_loader)*0.2)], generator)

train_loader = torch.utils.data.DataLoader(train_loader, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_loader, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=1, shuffle=False)


encoder = whisper.load_model("base").encoder
# encoder = torch.load("encoder_167.h5")

for param in encoder.parameters():
   param.requires_grad = False

for param in encoder.conv1.parameters():
    param.requires_grad = True

for param in encoder.conv2.parameters():
    param.requires_grad = True

reference_encoder = whisper.load_model("base").encoder

for param in reference_encoder.parameters():
  param.requires_grad = False

encoder = encoder.to(device)
reference_encoder = reference_encoder.to(device)

distance = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=encoder.parameters() ,lr=3e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
early_stopping = EarlyStopping(patience=5, path='best_encoder_check.h5')

for epoch in range(1,1000):
  encoder.train()
  train_loss = []
  for attacked_mel, original_mel in train_loader:

    attacked_embed = encoder(attacked_mel)
    original_new_embed = encoder(original_mel)
    original_embed = reference_encoder(original_mel)

    loss = distance(attacked_embed, original_embed) + distance(original_new_embed, original_embed)
    train_loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()

  encoder.eval()
  val_loss = []
  for attacked_mel, original_mel in val_loader:
    
    with torch.no_grad():
      attacked_embed = encoder(attacked_mel)
      original_new_embed = encoder(original_mel)
    original_embed = reference_encoder(original_mel)

    loss = distance(attacked_embed, original_embed) + distance(original_new_embed, original_embed)
    val_loss.append(loss.item())

  early_stopping(sum(val_loss)/len(val_loss), encoder)
  if early_stopping.early_stop:
    print("Early stopping")
    break
  print("epoch :", epoch, " train_loss :", sum(train_loss)/len(train_loss), " val_loss :", sum(val_loss)/len(val_loss))

encoder.eval()
test_loss = []
for attacked_mel, original_mel in test_loader:
  with torch.no_grad():
    attacked_embed = encoder(attacked_mel)
    original_new_embed = encoder(original_mel)
  original_embed = reference_encoder(original_mel)

  loss = distance(attacked_embed, original_embed) + distance(original_new_embed, original_embed)
  test_loss.append(loss.item())
print("test_loss :", sum(test_loss)/len(test_loss))