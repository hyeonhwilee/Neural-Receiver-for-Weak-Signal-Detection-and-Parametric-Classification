# Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification
A Multi-Task IQ-Based Neural Receiver for Weak-Signal Detection and Parametric Classification

## 프로젝트 소개
이 프로젝트는 약한 신호 검출과 파라미터 분류를 위한 멀티태스크 신경망 수신기를 구현합니다. IQ(In-phase/Quadrature) 신호를 입력으로 받아 다음 작업을 수행합니다:
- 약한 신호 검출 (Signal Detection)
- 신호 파라미터 분류 (Parametric Classification)

## 🚀 Google Colab에서 실행하기

> **💡 더 자세한 실행 가이드는 [QUICK_START.md](QUICK_START.md)를 참고하세요!**

### ⚡ 빠른 시작 (원클릭 실행)

**전체 시뮬레이션을 바로 실행하려면:**
- [run_simulation.ipynb](run_simulation.ipynb) - 전체 End-to-End 시뮬레이션
  - 데이터 생성, 모델 학습, 평가, 시각화를 모두 포함
  - 실행만 하면 모든 결과를 확인 가능!

**환경 설정만 필요하면:**
- [setup_colab.ipynb](setup_colab.ipynb) - 기본 환경 설정 및 예제

### 방법 1: 전체 시뮬레이션 노트북 실행 (권장)
다음 링크를 클릭하여 전체 시뮬레이션을 바로 실행하세요:
```
https://colab.research.google.com/github/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification/blob/main/run_simulation.ipynb
```

**포함된 내용:**
1. ✅ 환경 자동 설정 (패키지 설치, GPU 확인)
2. ✅ IQ 신호 데이터 생성 (다양한 변조 방식)
3. ✅ 멀티태스크 신경망 모델 학습
4. ✅ 성능 평가 및 시각화
5. ✅ SNR별 성능 분석 (-20dB ~ 20dB)

### 방법 2: 새 Colab 노트북에서 원클릭 코드 실행
새 Colab 노트북을 만들고 아래 코드를 **한 번에** 실행하세요:

```python
# 1. 저장소 클론
!git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
%cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# 2. 필요한 패키지 설치
!pip install -q -r requirements.txt

# 3. GPU 확인
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 4. 기본 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

print("✓ 환경 설정 완료!")
```

### GPU 설정
더 빠른 학습을 위해 GPU를 활성화하세요:
- 런타임 > 런타임 유형 변경 > 하드웨어 가속기 > **GPU** 선택

## 로컬 환경에서 실행하기

```bash
# 1. 저장소 클론
git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# 2. 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

## 주요 기능
- IQ 신호 생성 및 전처리
- 약한 신호 검출을 위한 신경망 모델
- 멀티태스크 학습 (검출 + 분류)
- 다양한 SNR(Signal-to-Noise Ratio) 환경 지원

## 필요한 패키지
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- SciPy >= 1.10.0
- 기타 (requirements.txt 참조)

## 📁 프로젝트 구조
```
.
├── run_simulation.ipynb    # 🌟 전체 시뮬레이션 노트북 (End-to-End)
├── setup_colab.ipynb       # Colab 환경 설정 노트북
├── requirements.txt        # 필요한 패키지 목록
└── README.md              # 프로젝트 설명
```

### 노트북 설명
- **run_simulation.ipynb**: 데이터 생성부터 학습, 평가까지 전체 과정을 포함한 완전한 시뮬레이션
  - 약 30 에폭 학습 (GPU 사용 시 약 5-10분 소요)
  - 신호 검출 및 변조 분류 성능 시각화
  - SNR별 성능 분석 그래프

- **setup_colab.ipynb**: 환경 설정 및 기본 예제 코드

## 기여
이슈와 Pull Request는 언제나 환영합니다!

## 라이센스
MIT License
