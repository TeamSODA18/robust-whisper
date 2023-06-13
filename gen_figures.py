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
    "TeamSODA/ae-signal_processing_attacks_whisper_librispeech", split="train"
)

loader = GetXYEval(train_data, device=device)

model = RobustWhisper()
original_model = whisper.load_model("base")
original_model = original_model.to(device)

encoder = whisper.load_model("base").encoder


attacked_list = []
original_list= []
i = 0

for attacked, original in loader:
    attacked_transcript_new = model(attacked.squeeze(), 16000)
    attacked_transcript_original = get_transcript(attacked, original_model)
    original_transcript_original = get_transcript(original, original_model)

    wer_attack, wer_recon = get_wer(
        original_transcript_original,
        attacked_transcript_original,
        attacked_transcript_new,
    )

    if wer_attack!=0:
        attacked_mel = get_mel(attacked.unsqueeze(0))
        original_mel = get_mel(original.unsqueeze(0))
        attacked_embed = encoder(attacked_mel)
        original_embed = encoder(original_mel)
        attacked_list.append(attacked_embed.detach().view(1,-1))
        original_list.append(original_embed.detach().view(1,-1))
        i+=1
    if i==10:
        break

both = original_list+attacked_list

both = torch.cat(both, dim=0).cpu()
both_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(both)
original_embedded = both_embedded[:both_embedded.shape[0]//2]
attacked_embedded = both_embedded[both_embedded.shape[0]//2:]
distances = np.sqrt(np.sum((original_embedded-attacked_embedded)**2, axis=1))
norm = matplotlib.colors.Normalize(vmin=np.min(distances), vmax=np.max(distances), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='seismic')
colors = mapper.to_rgba(distances)

for i in range(original_embedded.shape[0]):
    plt.plot([original_embedded[i,0], attacked_embedded[i,0]], 
            [original_embedded[i,1], attacked_embedded[i,1]],
            c = colors[i],
            )

    plt.scatter(attacked_embedded[i,0], 
            attacked_embedded[i,1],
            c = [colors[i]],
            marker = "X",
            label="attacked",
            )
    plt.scatter(original_embedded[i,0], 
            original_embedded[i,1],
            c = [colors[i]],
            marker = "o",
            label="original",
            )

legend_elements = [Line2D([0], [0], marker='X', color='w', label='attacked', markerfacecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='original', markerfacecolor='black'),
                   ]

plt.tick_params(left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)
plt.legend(handles=legend_elements)
plt.savefig("figures/latent_space.pdf")