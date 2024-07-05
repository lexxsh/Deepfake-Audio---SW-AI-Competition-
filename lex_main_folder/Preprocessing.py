import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import whisper
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


def update_paths(data, base_path):
    data['path'] = data['path'].apply(lambda p: p.replace('./', f'{base_path}'))
    return data

def load_and_split_data(csv_file, base_path='../SW/'):
    data = pd.read_csv(csv_file)
    data = update_paths(data, base_path)

    # 라벨이 'real'인 데이터를 필터링합니다.
    real_data = data[data['label'] == 'real']

    # 라벨이 'fake'인 데이터를 필터링합니다.
    fake_data = data[data['label'] == 'fake']

    # real_data와 fake_data의 길이를 맞추기 위해 최소 길이를 사용합니다.
    min_length = min(len(real_data), len(fake_data))

    # 각각 real_data와 fake_data에서 min_length만큼 샘플링하여 길이를 맞춥니다.
    real_data = real_data.sample(min_length).reset_index(drop=True)
    fake_data = fake_data.sample(min_length).reset_index(drop=True)

    # 랜덤 인덱스를 생성하여 데이터프레임을 셔플합니다.
    shuffled_real_indices = np.random.permutation(min_length)
    shuffled_fake_indices = np.random.permutation(min_length)

    # 랜덤하게 결합할 데이터프레임을 생성합니다.
    real_data_1 = real_data.iloc[shuffled_real_indices].reset_index(drop=True)
    real_data_2 = real_data.iloc[shuffled_fake_indices].reset_index(drop=True).add_suffix('.1')

    real_real = pd.concat([real_data_1, real_data_2], axis=1)
    
    fake_data_2 = fake_data.iloc[shuffled_fake_indices].reset_index(drop=True).add_suffix('.1')
    real_fake = pd.concat([real_data_1, fake_data_2], axis=1)
    
    fake_data_1 = fake_data.iloc[shuffled_real_indices].reset_index(drop=True)
    fake_fake = pd.concat([fake_data_1, fake_data_2], axis=1)

    return real_real, real_fake, fake_fake


def resample_audio(data, original_rate, target_rate):
    number_of_samples = round(len(data) * float(target_rate) / original_rate)
    resampled_data = resample(data, number_of_samples)
    return resampled_data

def concatenate_audios_overlap(file_path1, file_path2, output_path):
    data1, samplerate1 = sf.read(file_path1)
    data2, samplerate2 = sf.read(file_path2)
    if samplerate1 != samplerate2:
        if samplerate1 > samplerate2:
            data2 = resample_audio(data2, samplerate2, samplerate1)
            samplerate2 = samplerate1
        else:
            data1 = resample_audio(data1, samplerate1, samplerate2)
            samplerate1 = samplerate2
    if len(data1) > len(data2):
        data1 = data1[:len(data2)]
    else:
        data2 = data2[:len(data1)]
    combined = data1 + data2
    sf.write(output_path, combined, samplerate1)
    
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_audio_set(real_real, real_fake, fake_fake):
    create_dir_if_not_exists('combined_audio')

    for i, row in tqdm(real_real.iterrows(), total=real_real.shape[0], desc="Processing real-real pairs"):
        file_path1 = row['path']
        file_path2 = row['path.1']
        output_path = f'combined_audio/real_real/{i}.ogg'
        concatenate_audios_overlap(file_path1, file_path2, output_path)

    for i, row in tqdm(real_fake.iterrows(), total=real_fake.shape[0], desc="Processing real-fake pairs"):
        file_path1 = row['path']
        file_path2 = row['path.1']
        output_path = f'combined_audio/real_fake/{i}.ogg'
        concatenate_audios_overlap(file_path1, file_path2, output_path)

    for i, row in tqdm(fake_fake.iterrows(), total=fake_fake.shape[0], desc="Processing fake-fake pairs"):
        file_path1 = row['path']
        file_path2 = row['path.1']
        output_path = f'combined_audio/fake_fake/{i}.ogg'
        concatenate_audios_overlap(file_path1, file_path2, output_path)

def main(csv_file):
    real_real, real_fake, fake_fake = load_and_split_data(csv_file)
    create_audio_set(real_real, real_fake, fake_fake)

if __name__ == '__main__':
    csv_file = '../SW/train.csv'  # 실제 CSV 파일 경로를 여기에 설정합니다.
    main(csv_file)
