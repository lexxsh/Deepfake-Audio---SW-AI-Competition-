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
from tqdm import tqdm  # tqdm 임포트 추가
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
# 다른 모듈들의 임포트

SAMPLING_RATE = 16000
APPLY_NORMALIZATION = True
APPLY_TRIMMING = True
APPLY_PADDING = True
FRAMES_NUMBER = 480000
win_length = 400
hop_length = 160

SOX_SILENCE = [
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

def resample_audio(data, original_rate, target_rate):
    return librosa.resample(data, orig_sr=original_rate, target_sr=target_rate)

def apply_preprocessing(waveform, sample_rate):
    if sample_rate != SAMPLING_RATE:
        waveform = resample_audio(waveform, sample_rate, SAMPLING_RATE)
        sample_rate = SAMPLING_RATE

    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0, :]

    if APPLY_TRIMMING:
        waveform = librosa.effects.trim(waveform, top_db=20)[0]

    if APPLY_PADDING:
        waveform = apply_pad(waveform, FRAMES_NUMBER)

    return waveform, sample_rate

def apply_pad(waveform, cut):
    waveform_len = len(waveform)

    if waveform_len >= cut:
        return waveform[:cut]

    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = np.tile(waveform, num_repeats)[:cut]

    return padded_waveform

def concatenate_audios_overlap(data1, data2):
    min_length = min(len(data1), len(data2))
    combined = data1[:min_length] + data2[:min_length]
    return combined

def read_audio(file_path):
    data, sample_rate = librosa.load(file_path, sr=None)
    return data, sample_rate

class SimpleAudioDataset(Dataset):
    def __init__(self, real_real_dir, real_fake_dir, fake_fake_dir, noise_dir, csv_file, transform=None, return_meta=False):
        self.real_real_dir = real_real_dir
        self.real_fake_dir = real_fake_dir
        self.fake_fake_dir = fake_fake_dir
        self.noise_dir = noise_dir
        self.samples = pd.read_csv(csv_file)
        self.transform = transform
        self.return_meta = return_meta

        self.data_files = []
        self.labels = []

        self.load_and_combine_data(self.real_real_dir, [0, 1])
        self.load_and_combine_data(self.real_fake_dir, [1, 1])
        self.load_and_combine_data(self.fake_fake_dir, [1, 0])
        self.load_noise_data()

        for _, row in tqdm(self.samples.iterrows(), desc="Loading Samples", total=len(self.samples)):
            file_path = '../SW/' + row['path']
            if random.random() < 1:
                file_path = self.combine_with_noise(file_path)
            self.data_files.append(file_path)
            self.labels.append([1, 0] if row['label'] == 'fake' else [0, 1])

    def load_and_combine_data(self, folder_path, label):
        for filename in tqdm(os.listdir(folder_path), desc=f"Loading {os.path.basename(folder_path)}", total=len(os.listdir(folder_path))):
            if filename.endswith('.ogg'):
                file_path = os.path.join(folder_path, filename)
                if random.random() < 1:
                    file_path = self.combine_with_noise(file_path)
                self.data_files.append(file_path)
                self.labels.append(label)

    def load_noise_data(self):
        for filename in tqdm(os.listdir(self.noise_dir), desc="Loading Noise Data", total=len(os.listdir(self.noise_dir))):
            if filename.endswith('.ogg'):
                self.data_files.append(os.path.join(self.noise_dir, filename))
                self.labels.append([0, 0])

    def combine_with_noise(self, audio_file):
        # 노이즈 파일 선택
        noise_file = random.choice([f for f in os.listdir(self.noise_dir) if f.endswith('.ogg')])
        noise_path = os.path.join(self.noise_dir, noise_file)
        
        # 오디오 및 노이즈 데이터 읽기
        audio_data, audio_sr = read_audio(audio_file)
        noise_data, noise_sr = read_audio(noise_path)

        # 샘플링 레이트 맞추기
        if audio_sr != noise_sr:
            noise_data = resample_audio(noise_data, noise_sr, audio_sr)
            noise_sr = audio_sr

        # 오디오 결합
        combined_data = concatenate_audios_overlap(audio_data, noise_data)
        
        # 임시 파일을 저장할 폴더 경로
        temp_folder = "temp_files(spectorgram)"

        # 폴더가 없으면 생성
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # 원본 파일명에서 확장자를 제외한 이름 추출
        base_name = os.path.splitext(os.path.basename(audio_file))[0]

        # 새로운 임시 파일 경로 (.ogg 확장자 사용)
        temp_file = os.path.join(temp_folder, f"temp_{base_name}.ogg")

        # .ogg 형식으로 저장
        sf.write(temp_file, combined_data, audio_sr, format='ogg')
        
        return temp_file

    def __del__(self):
        for file in self.data_files:
            if file.startswith("temp_"):
                os.remove(file)
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        path = self.data_files[index]
        label = self.labels[index]

        waveform, sample_rate = read_audio(path)
        real_sec_length = len(waveform) / sample_rate
        waveform, sample_rate = apply_preprocessing(waveform, sample_rate)
        label = torch.tensor(label, dtype=torch.float)
        return_data = [waveform, sample_rate, label]
        if self.return_meta:
            file_id = os.path.basename(path).split('.')[0]
            return_data.append((file_id, path, real_sec_length))

if __name__ == '__main__':
    real_real_dir = './combined_audio2/real_real'
    real_fake_dir = './combined_audio2/real_fake'
    fake_fake_dir = './combined_audio2/fake_fake'
    noise_dir = './output_noise/'
    csv_file = '../SW/train.csv'

    dataset = SimpleAudioDataset(real_real_dir, real_fake_dir, fake_fake_dir, noise_dir, csv_file, return_meta=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 모델 초기화 및 학습 설정
    model_config = {
        'fc1_dim': 1024,
        'frontend_algorithm': ["mfcc_spectogram"],
        'input_channels': 1
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = meso_net.FrontendMesoInception4(fc1_dim=model_config['fc1_dim'],
                                            frontend_algorithm=model_config['frontend_algorithm'],
                                            input_channels=model_config['input_channels'],
                                            device=device)
    
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path, interval=3):
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
            best_model_path = os.path.join(save_path, f'best_model{epoch+1}.pth')
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
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * 2, T_mult=1, eta_min=5e-6)

save_path = 'checkpoints(augmentataion+_spectogram)'
os.makedirs(save_path, exist_ok=True)

epochs = 100
train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path)
