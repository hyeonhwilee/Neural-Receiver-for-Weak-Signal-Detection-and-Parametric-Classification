# Google Colab 빠른 시작 가이드

Google Colab에서 Neural Receiver 시뮬레이션을 실행하는 완벽한 가이드입니다.

## 🚀 방법 1: 원클릭 실행 (추천)

Google Colab에서 새 노트북을 열고 아래 코드를 **하나의 셀에 복사해서 실행**하세요:

```python
# ========================================
# Neural Receiver 전체 시뮬레이션 실행
# ========================================

# 1. 저장소 클론
!git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
%cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# 2. 올바른 브랜치로 체크아웃
!git checkout claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc

# 3. 최신 코드 가져오기
!git pull origin claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc

# 4. 필요한 패키지 설치
!pip install -q torch torchvision tensorboard numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml nbformat nbconvert

# 5. run_simulation.ipynb를 Python 스크립트로 변환하여 실행
!jupyter nbconvert --to script run_simulation.ipynb --output run_simulation_exec
!python run_simulation_exec.py
```

**예상 실행 시간**:
- CPU: 15-20분
- GPU (T4): 8-12분

---

## 🎯 방법 2: 단계별 실행

각 단계를 별도의 셀에서 실행하면서 진행 상황을 확인하고 싶다면 이 방법을 사용하세요.

### Step 1: 환경 설정

```python
# 저장소 클론 및 이동
!git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
%cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# 올바른 브랜치로 체크아웃
!git checkout claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc
!git pull origin claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc
```

### Step 2: 패키지 설치

```python
# 필수 패키지 설치
!pip install -q torch torchvision tensorboard numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml
```

### Step 3: GPU 확인 (선택사항)

```python
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전: {torch.version.cuda}")
```

### Step 4: 시뮬레이션 실행

```python
# run_simulation.ipynb를 Python 스크립트로 변환
!pip install -q nbformat nbconvert
!jupyter nbconvert --to script run_simulation.ipynb --output run_simulation_exec

# Python 스크립트 실행
!python run_simulation_exec.py
```

### Step 5: 결과 확인

```python
# 생성된 파일 목록 확인
!ls -lh *.png *.json

# 결과 이미지 표시
from IPython.display import Image, display
import os

result_images = [
    'signal_examples.png',
    'training_history.png',
    'detection_metrics.png',
    'confusion_matrix.png',
    'regression_errors.png'
]

for img in result_images:
    if os.path.exists(img):
        print(f"\n{'='*70}")
        print(f"📊 {img}")
        print('='*70)
        display(Image(filename=img))
```

### Step 6: 결과 다운로드

```python
# 결과를 ZIP으로 압축하여 다운로드
!zip -r neural_receiver_results.zip *.png *.json experiments/neural_receiver/checkpoints/*.pt

from google.colab import files
files.download('neural_receiver_results.zip')
```

---

## 🎨 방법 3: 노트북 직접 실행

Colab에서 `run_simulation.ipynb`를 직접 열어서 대화형으로 실행하고 싶다면:

### 옵션 A: GitHub에서 직접 열기

다음 링크를 클릭하여 Colab에서 바로 열기:

```
https://colab.research.google.com/github/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification/blob/claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc/run_simulation.ipynb
```

### 옵션 B: 파일 업로드

1. 로컬에 저장소를 클론:
   ```bash
   git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
   cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification
   git checkout claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc
   ```

2. `run_simulation.ipynb` 파일을 Google Colab에 업로드

3. 각 셀을 순차적으로 실행

---

## 📋 실행 전 체크리스트

- [ ] Google Colab 계정이 있는지 확인
- [ ] (선택) GPU 런타임 활성화: `런타임` → `런타임 유형 변경` → `GPU` 선택
- [ ] 충분한 시간 확보 (15-20분)

---

## 🔧 GPU 활성화 방법 (권장)

GPU를 사용하면 훈련 속도가 2-3배 빨라집니다:

1. Colab 메뉴에서 `런타임` → `런타임 유형 변경` 클릭
2. `하드웨어 가속기`를 `GPU`로 변경
3. `저장` 클릭

---

## 📊 예상 결과

시뮬레이션이 성공적으로 완료되면 다음 파일들이 생성됩니다:

### 생성되는 파일
- `signal_examples.png` - 6가지 변조 타입 신호 예시
- `training_history.png` - 훈련 과정 그래프
- `detection_metrics.png` - ROC 곡선 및 Precision-Recall 곡선
- `confusion_matrix.png` - 변조 분류 혼동 행렬
- `regression_errors.png` - 파라미터 추정 오차
- `simulation_results.json` - 전체 메트릭 결과
- `experiments/neural_receiver/checkpoints/best_model.pt` - 최고 성능 모델

### 예상 성능 (SNR ≤ 0 dB)
- **Detection ROC-AUC**: > 0.95
- **Classification Accuracy**: > 0.85
- **Regression MAE**: < 0.1

---

## ⚠️ 문제 해결

### 문제: "git: command not found"
**해결**: Colab에는 기본적으로 git이 설치되어 있습니다. 런타임을 재시작하세요.

### 문제: "CUDA out of memory"
**해결**:
```python
# run_simulation_exec.py를 수정하기 전에 다음 코드를 실행
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```
또는 batch_size를 줄이세요.

### 문제: "패키지 설치 실패"
**해결**:
```python
# 캐시를 무시하고 재설치
!pip install --no-cache-dir torch torchvision tensorboard
```

### 문제: "run_simulation.ipynb 파일이 없습니다"
**해결**:
```python
# 브랜치를 정확히 체크아웃했는지 확인
!git branch
!git checkout claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc
!ls -la run_simulation.ipynb
```

---

## 📞 도움말

더 많은 정보는 다음 문서를 참조하세요:
- [README.md](README.md) - 프로젝트 개요
- [QUICKSTART.md](QUICKSTART.md) - 로컬 환경 설정 가이드
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 상세 프로젝트 설명

---

## 🎓 다음 단계

시뮬레이션이 완료된 후:

1. **결과 분석**: `simulation_results.json` 파일을 열어 상세 메트릭 확인
2. **모델 사용**: 저장된 모델로 새로운 신호에 대해 추론 수행
3. **하이퍼파라미터 튜닝**: `configs/default_config.yaml` 수정하여 재훈련
4. **커스텀 신호**: `src/data/signal_generator.py`에 새로운 변조 타입 추가

---

## 💡 팁

- 첫 실행 시 패키지 설치에 1-2분 소요됩니다.
- GPU를 사용하면 훈련 시간이 크게 단축됩니다.
- 중간 결과를 확인하려면 방법 2(단계별 실행)를 권장합니다.
- 결과 이미지는 Colab 파일 브라우저에서도 볼 수 있습니다.

---

**Happy Signal Processing! 📡**
