## SW 2024 경진대회


## lexxsh
### 데이터 증강(real_real, real_fake, fake_fake 음성 두개를 합치는 증강 기법) [lex_main/Preprocessing.py] => 진행 후 combined/folder에 저장
### Mesonet + MFCC [lex_main/Pretrained.ipynb] ==> 0.31514
### Mesonet + MFCC + data 증강(real_real, fake_fake, real_fake 데이터셋 추가) [lex_main/main.py] ==> 학습중

## jongmin
### Resnet + MFCC (base.ipynb) : 0.4543810021
### attention+transformer+MFCC:  0.4670262771
### attention+transformer+MFCC (augumentation 2개음성): 0.4395052897
### MFCC+spectogram+CNN+LSTM(augumentation 노이즈적용+2개음성 ):0.4131263582   
