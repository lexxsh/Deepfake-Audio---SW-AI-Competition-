import torchaudio
import noisereduce as nr
import os
import torch

# 저장할 디렉터리 설정
save_dir = 'output'
os.makedirs(save_dir, exist_ok=True)

# 오디오 파일 로드 및 전처리
print("오디오 파일을 로드 중입니다...")
waveform, sr = torchaudio.load('../SW/unlabeled_data/ASAPHKOM.ogg')
waveform = waveform.numpy()  # To use noisereduce, convert to numpy array
print("오디오 파일 로드 완료.")

# 만약 모노 오디오인 경우
if waveform.shape[0] == 1:
    waveform_stereo = waveform.repeat(2, axis=0)  # 두 개의 채널로 확장
else:
    waveform_stereo = waveform

print("노이즈 제거 중...")
# 노이즈 제거 처리
reduced_noise = nr.reduce_noise(y=waveform_stereo, sr=sr)
print("노이즈 제거 완료.")

# 다시 텐서로 변환
reduced_noise_tensor = torch.from_numpy(reduced_noise)

# 모노 오디오로 변환 (원래 모노였을 경우)
if reduced_noise_tensor.shape[0] == 2:
    reduced_noise_tensor_mono = reduced_noise_tensor.mean(dim=0, keepdim=True)
else:
    reduced_noise_tensor_mono = reduced_noise_tensor

# 결과 저장 경로 지정
cleaned_audio_path = os.path.join(save_dir, 'cleaned_audio.wav')

# 오디오 파일 저장
torchaudio.save(cleaned_audio_path, reduced_noise_tensor_mono, sr)
print(f"변환된 오디오 파일이 '{save_dir}' 디렉터리에 저장되었습니다.")
