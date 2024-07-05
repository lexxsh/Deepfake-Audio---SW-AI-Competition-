import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import random
import os
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.signal import resample
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.models import (
    lcnn,
    specrnet,
    whisper_specrnet,
    rawnet3,
    whisper_lcnn,
    meso_net,
    whisper_meso_net
)
import warnings
warnings.filterwarnings('ignore')


SAMPLING_RATE = 16_000
APPLY_NORMALIZATION = True
APPLY_TRIMMING = True
APPLY_PADDING = True
FRAMES_NUMBER = 480_000
win_length = 400
hop_length = 160

SOX_SILENCE = [
    # Trim silence longer than 0.2s and louder than 1% volume
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

SOX_SILENCE = [
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

def resample_audio(data, original_rate, target_rate):
    number_of_samples = round(len(data) * float(target_rate) / original_rate)
    resampled_data = resample(data, number_of_samples)
    return resampled_data

def apply_preprocessing(waveform, sample_rate):
    if sample_rate != SAMPLING_RATE:
        waveform, sample_rate = resample_wave(waveform, sample_rate, SAMPLING_RATE)

    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform[:1, ...]

    if APPLY_TRIMMING:
        waveform, sample_rate = apply_trim(waveform, sample_rate)

    if APPLY_PADDING:
        waveform = apply_pad(waveform, FRAMES_NUMBER)
    return waveform, sample_rate


def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    return waveform, sample_rate


def apply_trim(waveform, sample_rate):
    waveform_trimmed, sample_rate_trimmed = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, SOX_SILENCE
    )

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed
    
    return waveform, sample_rate


def apply_pad(waveform, cut):
    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    return padded_waveform




class SimpleAudioDataset(Dataset):
    def __init__(self, real_real_dir, real_fake_dir, fake_fake_dir, csv_file, transform=None, return_meta=False):
        self.real_real_dir = real_real_dir
        self.real_fake_dir = real_fake_dir
        self.fake_fake_dir = fake_fake_dir
        self.samples = pd.read_csv(csv_file)
        self.transform = transform
        self.return_meta = return_meta

        self.data_files = []
        self.labels = []
        
        # 폴더에서 파일 읽기
        self.load_folder_data(self.real_real_dir, [0, 1])
        self.load_folder_data(self.real_fake_dir, [1, 1])
        self.load_folder_data(self.fake_fake_dir, [1, 0])

        # CSV에서 파일 읽기
        for _, row in self.samples.iterrows():
            self.data_files.append('../SW/' + row['path'])
            self.labels.append([1, 0] if row['label'] == 'fake' else [0, 1])

    def load_folder_data(self, folder_path, label):
        for filename in os.listdir(folder_path):
            if filename.endswith('.ogg'):
                self.data_files.append(os.path.join(folder_path, filename))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):
        path = self.data_files[index]
        label = self.labels[index]

        waveform, sample_rate = torchaudio.load(path, normalize=APPLY_NORMALIZATION)
        real_sec_length = len(waveform[0]) / sample_rate

        waveform, sample_rate = apply_preprocessing(waveform, sample_rate)
        label = torch.tensor(label, dtype=torch.float)
        return_data = [waveform, sample_rate, label]
        if self.return_meta:
            file_id = os.path.basename(path).split('.')[0]
            return_data.append((file_id, path, real_sec_length))

        return return_data

if __name__ == '__main__':
    real_real_dir = './combined_audio/real_real'
    real_fake_dir = './combined_audio/real_fake'
    fake_fake_dir = './combined_audio/fake_fake'
    csv_file = '../SW/train.csv'

    dataset = SimpleAudioDataset(real_real_dir, real_fake_dir, fake_fake_dir, csv_file, return_meta=True)
    train_size = int(0.8 * len(dataset))  # 80% of data for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoader for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    
    
model_config = {
    'fc1_dim': 1024,
    'frontend_algorithm': ["mfcc"],
    'input_channels': 1
}




device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = meso_net.FrontendMesoInception4(fc1_dim=model_config['fc1_dim'],
                       frontend_algorithm=model_config['frontend_algorithm'],
                       input_channels=model_config['input_channels'],
                       device=device)



from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path, interval=5):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")
        
        for batch in train_progress:
            waveforms, sample_rates, labels, *meta = batch
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate

            running_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item()})

        average_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {average_loss:.4f}")

        # 검증 단계
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

        # 최고의 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")
        
        # 매 interval (기본값: 5) 에폭마다 모델 저장
        if (epoch + 1) % interval == 0:
            interval_model_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), interval_model_path)
            print(f"Model saved at {interval_model_path}")

def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    val_progress = tqdm(val_loader, desc="Validating")
    
    with torch.no_grad():
        for batch in val_progress:
            waveforms, sample_rates, labels, *meta = batch
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            val_progress.set_postfix({'loss': loss.item()})

    average_val_loss = total_val_loss / len(val_loader)
    return average_val_loss

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

save_path = 'checkpoints'
os.makedirs(save_path, exist_ok=True)

epochs = 100
train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path)