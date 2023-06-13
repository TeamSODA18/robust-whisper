from spClassify.infer import SpMelCl
from robust_whisper.models import RobustWhisper
import torchaudio
import torch
from datasets import load_dataset
from denoiser import pretrained
from denoiser.dsp import convert_audio
from utils import *
from speechbrain.pretrained import SpectralMaskEnhancement

# enhance_model = SpectralMaskEnhancement.from_hparams(
#     source="speechbrain/metricgan-plus-voicebank",
#     savedir="pretrained_models/metricgan-plus-voicebank",
# )

enhance_model = pretrained.dns64().cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = load_dataset(
    # "TeamSODA/ae-signal_processing_attacks_whisper_librispeech", split="train"
    # "TeamSODA/ae-signal_processing_attacks_whisper_commonvoice", split="train"
    "TeamSODA/ae-signal_processing_attacks_assembly_librispeech", split="train"
    #"TeamSODA/ae-signal_processing_attacks_assembly_commonvoice", split="train"
)
loader = GetXYEval(train_data, device=device)
# generator = torch.Generator().manual_seed(42)
# _, _, loader = torch.utils.data.random_split(
#     loader,
#     [
#         int(len(loader) * 0.6),
#         int(len(loader) * 0.2),
#         int(len(loader) * 0.2),
#     ],
#     generator,
# )

original_model = whisper.load_model("base")
original_model = original_model.to(device)

original_defended = []

for attacked, original in loader:
    original_text = get_transcript(original, original_model, device)
    
    # enhanced = enhance_model.enhance_batch(original.unsqueeze(0), lengths=torch.tensor([1.]))
    wav = convert_audio(original.unsqueeze(0).cuda(), 16000, enhance_model.sample_rate, enhance_model.chin)
    with torch.no_grad():
        enhanced = enhance_model(wav[None])[0]
    
    defended_text = get_transcript(enhanced.squeeze(), original_model, device)
    
    wer_recon = get_wer_single(original_text, defended_text)
    original_defended.append(wer_recon)
    print("benign", wer_recon)

print("original defended wer", sum(original_defended)/len(original_defended))