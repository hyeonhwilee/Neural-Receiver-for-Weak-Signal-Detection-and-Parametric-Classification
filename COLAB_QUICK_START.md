# Google Colab 빠른 시작 가이드

Google Colab에서 아래 코드를 복사해서 셀에 붙여넣고 실행하세요.

## 방법 1: 한 번에 전체 실행 (추천)

새 코드 셀을 만들고 아래 코드를 붙여넣은 후 실행(Shift+Enter):

```python
# ============================================================================
# SIGINT Neural Receiver - Google Colab 전체 시뮬레이션
# ============================================================================

print("="*70)
print("🚀 SIGINT Receiver Simulation - Google Colab")
print("="*70)

# [1/5] GPU 확인
print("\n[1/5] GPU 확인...")
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# [2/5] 저장소 클론
print("\n[2/5] 저장소 클론...")
import os
import shutil

os.chdir('/content')
if os.path.exists('Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification'):
    shutil.rmtree('Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification')

# 브랜치 지정 클론
BRANCH = "claude/fix-sigint-simulation-errors-01GfBwMpqJRhjxy1LkgE4GfF"
!git clone -b {BRANCH} https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git

os.chdir('/content/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification')
print(f"✓ 디렉토리: {os.getcwd()}")

# [3/5] 라이브러리 설치
print("\n[3/5] 라이브러리 설치...")
!pip install -q numpy scipy matplotlib torch torchvision scikit-learn pandas seaborn

import torch
print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")

# [4/5] 시뮬레이션 실행
print("\n[4/5] 시뮬레이션 실행... (약 2-5분)")
!python sigint_simulation.py

# [5/5] 결과 표시
print("\n[5/5] 결과 표시...")
from IPython.display import Image, display

print("\n" + "="*70)
print("📊 시뮬레이션 결과")
print("="*70)

print("\n1️⃣ 샘플 Spectrogram (Sine, Chirp, FSK, FHSS):")
display(Image('sample_spectrograms.png'))

print("\n2️⃣ 학습 결과 (손실 및 정확도):")
display(Image('training_results.png'))

print("\n3️⃣ Confusion Matrix:")
display(Image('confusion_matrix.png'))

print("\n✅ 완료! 생성된 파일:")
print("  - sample_spectrograms.png")
print("  - training_results.png")
print("  - confusion_matrix.png")
print("  - sigint_detector_model.pth")

# 결과 다운로드 (선택사항)
print("\n💾 파일을 다운로드하려면 아래 코드의 주석을 해제하세요:")
print("# from google.colab import files")
print("# files.download('sample_spectrograms.png')")
print("# files.download('training_results.png')")
print("# files.download('confusion_matrix.png')")
print("# files.download('sigint_detector_model.pth')")
```

---

## 방법 2: 단계별 실행

각 단계를 별도의 셀로 나눠서 실행하고 싶다면:

### 셀 1: 환경 설정
```python
print("="*70)
print("🚀 SIGINT Receiver Simulation")
print("="*70)

print("\n[1/5] GPU 확인...")
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

### 셀 2: 저장소 클론
```python
print("\n[2/5] 저장소 클론...")
import os
import shutil

os.chdir('/content')
if os.path.exists('Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification'):
    shutil.rmtree('Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification')

BRANCH = "claude/fix-sigint-simulation-errors-01GfBwMpqJRhjxy1LkgE4GfF"
!git clone -b {BRANCH} https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git

os.chdir('/content/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification')
print(f"현재 디렉토리: {os.getcwd()}")
!git branch
```

### 셀 3: 라이브러리 설치
```python
print("\n[3/5] 라이브러리 설치...")
!pip install -q numpy scipy matplotlib torch torchvision scikit-learn pandas seaborn

import torch
print(f"\n✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
```

### 셀 4: 시뮬레이션 실행
```python
print("\n[4/5] 시뮬레이션 실행... (약 2-5분)")
!python sigint_simulation.py
```

### 셀 5: 결과 표시
```python
print("\n[5/5] 결과 표시...")
from IPython.display import Image, display

print("\n" + "="*70)
print("📊 시뮬레이션 결과")
print("="*70)

print("\n1️⃣ 샘플 Spectrogram (Sine, Chirp, FSK, FHSS):")
display(Image('sample_spectrograms.png'))

print("\n2️⃣ 학습 결과 (손실 및 정확도):")
display(Image('training_results.png'))

print("\n3️⃣ Confusion Matrix:")
display(Image('confusion_matrix.png'))

print("\n✅ 완료! 생성된 파일:")
print("  - sample_spectrograms.png")
print("  - training_results.png")
print("  - confusion_matrix.png")
print("  - sigint_detector_model.pth")
```

### 셀 6 (선택): 결과 다운로드
```python
from google.colab import files
files.download('sample_spectrograms.png')
files.download('training_results.png')
files.download('confusion_matrix.png')
files.download('sigint_detector_model.pth')
```

---

## 주의사항

1. **GPU 런타임 설정**: 런타임 > 런타임 유형 변경 > T4 GPU 선택
2. **실행 시간**: 전체 시뮬레이션은 약 2-5분 소요됩니다
3. **메모리**: 최소 2GB GPU 메모리 필요
4. **브랜치 변경**: main 브랜치를 사용하려면 `BRANCH = "main"`으로 변경하세요

## 문제 해결

- **파일을 찾을 수 없음**: 저장소 클론이 완료되었는지 확인
- **GPU 없음**: 런타임 유형을 GPU로 변경
- **메모리 부족**: 런타임을 재시작하고 다시 시도
