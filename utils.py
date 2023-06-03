import whisper
import torch


def get_mel(X):
    X = whisper.pad_or_trim(X)
    X = whisper.log_mel_spectrogram(X)
    return X


class GetXY(Dataset):
    def __init__(self, dataset: Dataset, device, shuffle: bool = True):
        super(GetXY, self).__init__()
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        attacked = torch.tensor(sample["audio_0"]["array"], dtype=torch.float32).to(
            self.device
        )
        original = torch.tensor(sample["audio_1"]["array"], dtype=torch.float32).to(
            self.device
        )
        attacked_mel = get_mel(attacked)
        original_mel = get_mel(original)
        return attacked_mel, original_mel


class EarlyStopping:
    def __init__(self, patience=10, delta=0, path="checkpoint.h5"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.path))

def get_wer(original_text, attacked_text, reconstructed_text):
    metric = WordErrorRate()
    metric.update([attacked_text], [original_text])
    wer_attack = metric.compute().item()

    metric = WordErrorRate()
    metric.update([reconstructed_text], [original_text])
    wer_recon = metric.compute().item()

    return wer_attack, wer_recon

def get_transcript(X, model):
    # detect the spoken language
    mel = get_mel(X)
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    return result.text