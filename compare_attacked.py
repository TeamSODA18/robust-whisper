from spClassify.infer import SpMelCl
from robust_whisper.models import RobustWhisper
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torch
from datasets import load_dataset
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
    "TeamSODA/ae-signal_processing_attacks_whisper_commonvoice", split="train"
    # "TeamSODA/ae-signal_processing_attacks_assembly_librispeech", split="train"
    # "TeamSODA/ae-signal_processing_attacks_assembly_commonvoice", split="train"
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

attacked_defended = []
attacked_undefended = []

for attacked, original in loader:
    attacked_transcript_original = get_transcript(attacked, original_model, device)
    original_transcript_original = get_transcript(original, original_model, device)
    label = bool(get_wer_single(original_transcript_original, attacked_transcript_original))
    
    if not label:
        continue
    
    #enhanced = enhance_model.enhance_batch(attacked.unsqueeze(0), lengths=torch.tensor([1.]))
    wav = convert_audio(attacked.unsqueeze(0).cuda(), 16000, enhance_model.sample_rate, enhance_model.chin)
    with torch.no_grad():
        enhanced = enhance_model(wav[None])[0]
    
    attacked_transcript_new = get_transcript(enhanced.squeeze(), original_model, device)
    
    wer_attack, wer_recon = get_wer(
        original_transcript_original,
        attacked_transcript_original,
        attacked_transcript_new,
    )
    
    #if label:
    attacked_undefended.append(wer_attack)
    attacked_defended.append(wer_recon)
    print("attacked", wer_attack, "->" ,wer_recon)

print("attacked undefended wer", sum(attacked_undefended)/len(attacked_undefended))
print("attacked defended wer", sum(attacked_defended)/len(attacked_defended))