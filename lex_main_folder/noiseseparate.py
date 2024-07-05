import os
import glob
import torchaudio
import torch
from tqdm import tqdm
from demucs import pretrained
from demucs.apply import apply_model

# 오디오 파일이 있는 폴더 경로 설정
audio_folder = '../SW/unlabeled_data'

# 출력 디렉터리 설정
save_dir = './output_noise'
os.makedirs(save_dir, exist_ok=True)

# Pre-trained Demucs 모델 로드
print("모델을 로딩 중입니다...")
demucs = pretrained.get_model('htdemucs')
print("모델 로딩 완료.")

# 폴더 내 모든 WAV 파일 처리
audio_files = glob.glob(os.path.join(audio_folder, '*.ogg'))

for audio_file in tqdm(audio_files, desc="Processing Files", unit="file"):
    waveform, sr = torchaudio.load(audio_file)
    
    # 모노 오디오를 감지하면 스테레오로 변환
    if waveform.size(0) == 1:
        waveform = waveform.repeat(2, 1)

    waveform = waveform.unsqueeze(0)  # 배치 차원 추가

    # 음성과 노이즈 분리
    sources = apply_model(demucs, waveform, device='cpu')
    noise = sources[0, 1]  # 분리된 노이즈 트랙

    # 스테레오 노이즈 트랙을 모노로 변환
    noise_mono = torch.mean(noise, dim=0, keepdim=True)

    # 파일 이름 추출 및 저장 경로 설정
    file_name = os.path.basename(audio_file)
    noise_file_name = f"mono_noise_{file_name}"
    noise_path = os.path.join(save_dir, noise_file_name)

    # 모노 노이즈 트랙 저장
    torchaudio.save(noise_path, noise_mono, sr)

print("모든 파일 처리가 완료되었습니다.")
