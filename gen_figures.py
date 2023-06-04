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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset(
    "TeamSODA/ae-signal_processing_attacks_whisper_librispeech", split="train"
)
train_loader = GetXY(train_data, device=device)
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
train_loader = torch.utils.data.DataLoader(train_loader, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_loader, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=1, shuffle=False)

encoder = whisper.load_model("base").encoder

attacked = []
original= []
i = 0
for attacked_mel, original_mel in train_loader:
    attacked_embed = encoder(attacked_mel)
    original_embed = encoder(original_mel)
    attacked.append(attacked_embed.detach().view(1,-1))
    original.append(original_embed.detach().view(1,-1))
    i+=1
    if i==2:
        break

both = original+attacked

both = torch.cat(both, dim=0).cpu()
both_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(both)
original_embedded = both_embedded[:both_embedded.shape[0]//2]
attacked_embedded = both_embedded[both_embedded.shape[0]//2:]
distances = np.sqrt(np.sum((original_embedded-attacked_embedded)**2, axis=0))
norm = matplotlib.colors.Normalize(vmin=np.min(distances), vmax=np.max(distances), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='inferno')
colors = mapper.to_rgba(distances)

for i in range(original_embedded.shape[0]):
    plt.plot([original_embedded[i,0], attacked_embedded[i,0]], 
            [original_embedded[i,1], attacked_embedded[i,1]],
            c = colors[i]
            )
    plt.scatter([original_embedded[i,0], attacked_embedded[i,0]], 
            [original_embedded[i,1], attacked_embedded[i,1]],
            c = colors[i])
plt.savefig("latent_space.pdf")