# 🚀 Google Colab 완전 실행 가이드

Neural Receiver 시뮬레이션을 Google Colab에서 바로 실행하는 **완전하고 테스트된** 가이드입니다.

---

## ⚡ 원클릭 실행 (추천)

Google Colab에서 **새 노트북**을 열고 아래 코드를 **하나의 셀에 복사해서 실행**하세요:

```python
# ========================================
# Neural Receiver 전체 시뮬레이션 실행
# ========================================

# 1. 저장소 클론 및 이동
!git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
%cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# 2. 올바른 브랜치로 체크아웃
!git checkout claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc

# 3. 필요한 패키지 설치
!pip install -q torch torchvision tensorboard numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml

# 4. 전체 시뮬레이션 실행
!python run_complete_simulation.py
```

### ⏱️ 예상 실행 시간
- **GPU (T4)**: 8-12분
- **GPU (A100)**: 5-8분
- **CPU only**: 15-20분

> 💡 **팁**: GPU를 활성화하면 2-3배 빠릅니다!
> - Colab 메뉴: `런타임` → `런타임 유형 변경` → `GPU` 선택

---

## 📊 결과 확인

시뮬레이션이 완료되면 다음 셀에서 결과를 확인하세요:

```python
# 결과 이미지 표시
from IPython.display import Image, display
import json
import os

result_images = [
    ('signal_examples.png', '신호 예제 (6가지 변조 타입)'),
    ('training_history.png', '훈련 과정'),
    ('detection_metrics.png', 'ROC & PR 곡선'),
    ('confusion_matrix.png', '혼동 행렬'),
    ('regression_errors.png', '파라미터 추정 오차')
]

for img_file, description in result_images:
    if os.path.exists(img_file):
        print(f"\n{'='*70}")
        print(f"📊 {description}")
        print('='*70)
        display(Image(filename=img_file))
    else:
        print(f"⚠️  {img_file} 파일을 찾을 수 없습니다.")

# 성능 메트릭 출력
print("\n" + "="*70)
print("🎯 최종 성능 메트릭")
print("="*70)

if os.path.exists('simulation_results.json'):
    with open('simulation_results.json', 'r') as f:
        results = json.load(f)

    print(f"\n📈 검출 성능:")
    print(f"  ROC-AUC:              {results['metrics']['detection']['roc_auc']:.4f}")
    print(f"  Accuracy:             {results['metrics']['detection']['accuracy']:.4f}")
    print(f"  F1 Score:             {results['metrics']['detection']['f1']:.4f}")

    print(f"\n🎯 분류 성능:")
    print(f"  Accuracy:             {results['metrics']['classification']['accuracy']:.4f}")
    print(f"  F1 Macro:             {results['metrics']['classification']['f1_macro']:.4f}")
    print(f"  F1 Weighted:          {results['metrics']['classification']['f1_weighted']:.4f}")

    print(f"\n📐 회귀 성능:")
    print(f"  MAE:                  {results['metrics']['regression']['mae']:.4f}")
    print(f"  RMSE:                 {results['metrics']['regression']['rmse']:.4f}")

    print("\n" + "="*70)
else:
    print("⚠️  simulation_results.json 파일을 찾을 수 없습니다.")
```

---

## 💾 결과 다운로드

결과를 ZIP으로 압축하여 다운로드:

```python
# 모든 결과를 ZIP으로 압축
!zip -r neural_receiver_results.zip \
    *.png \
    *.json \
    experiments/neural_receiver/checkpoints/best_model.pt \
    experiments/neural_receiver/checkpoints/last_model.pt

# 다운로드
from google.colab import files
files.download('neural_receiver_results.zip')

print("✓ 다운로드 완료!")
```

---

## 🔬 개별 신호 테스트

학습된 모델로 개별 신호를 테스트하려면:

```python
import numpy as np
import torch
from src.data import SignalGenerator
from src.models import NeuralReceiver

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = NeuralReceiver(
    input_channels=2,
    base_channels=64,
    num_blocks=4,
    num_classes=12,
    num_params=4,
    dropout=0.3
).to(device)

checkpoint = torch.load('experiments/neural_receiver/checkpoints/best_model.pt',
                       map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 테스트 신호 생성
from src.data.signal_generator import SignalParams

signal_gen = SignalGenerator(seed=42)

# 특정 변조 타입의 파라미터 직접 생성
test_params = SignalParams(
    center_freq=0.1,
    bandwidth=0.05,
    power=1.0,
    snr_db=-8.0,
    symbol_rate=0.05,  # 정규화된 심볼레이트 (0.01 ~ 0.2 권장)
    modulation_type='FM'  # 여기를 변경하여 다른 변조 타입 테스트
)
_, test_signal = signal_gen.generate_signal(1024, test_params)

# 추론
iq_tensor = torch.stack([
    torch.from_numpy(test_signal.real).float(),
    torch.from_numpy(test_signal.imag).float()
]).unsqueeze(0).to(device)

with torch.no_grad():
    predictions = model.predict(iq_tensor)

# 결과 출력
modulation_classes = ['No-signal', 'AM', 'FM', '2FSK', 'CW Radar', 'Pulsed Radar',
                     'LFM/Chirp', 'Phase-Coded Pulse', 'Frequency-Coded Pulse',
                     'Unmodulated Pulse', 'Pulse-Compression MOP', 'Other MOP Types']

print("\n" + "="*70)
print("🔍 추론 결과")
print("="*70)
print(f"신호 검출:        {predictions['signal_detected'].item()}")
print(f"검출 확률:        {predictions['detection_prob'].item():.4f}")
print(f"변조 타입:        {modulation_classes[predictions['modulation_class'].item()]}")
print(f"분류 확신도:      {predictions['classification_confidence'].item():.4f}")
print(f"\n추정 파라미터:")
print(f"  Center Freq:    {predictions['parameters'][0, 0].item():.4f}")
print(f"  Bandwidth:      {predictions['parameters'][0, 1].item():.4f}")
print(f"  SNR:            {predictions['parameters'][0, 2].item():.2f} dB")
print(f"  Symbol Rate:    {predictions['parameters'][0, 3].item():.1f}")
print("="*70)
```

---

## 🎨 단계별 실행 (선택사항)

전체 과정을 단계별로 확인하고 싶다면:

### Step 1: 환경 설정
```python
!git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
%cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification
!git checkout claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc
!pip install -q torch torchvision tensorboard numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml
```

### Step 2: GPU 확인
```python
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 3: 시뮬레이션 실행
```python
!python run_complete_simulation.py
```

### Step 4: 결과 확인
```python
# 위의 "결과 확인" 섹션 코드 사용
```

---

## 📁 생성되는 파일

시뮬레이션 완료 후 다음 파일들이 생성됩니다:

| 파일 | 설명 |
|------|------|
| `signal_examples.png` | 6가지 변조 타입의 신호 예제 (시간/주파수 영역) |
| `training_history.png` | 훈련 과정 (손실, 정확도, MAE) |
| `detection_metrics.png` | ROC 곡선 & Precision-Recall 곡선 |
| `confusion_matrix.png` | 12x12 변조 분류 혼동 행렬 |
| `regression_errors.png` | 4개 파라미터 추정 오차 플롯 |
| `simulation_results.json` | 모든 메트릭 결과 (JSON) |
| `experiments/neural_receiver/checkpoints/best_model.pt` | 최고 성능 모델 |
| `experiments/neural_receiver/checkpoints/last_model.pt` | 마지막 에포크 모델 |
| `experiments/neural_receiver/training_history.json` | 에포크별 훈련 히스토리 |

---

## 🎯 예상 성능

약한 신호 조건 (SNR ≤ 0 dB)에서:

- **Detection ROC-AUC**: > 0.95 (신호 검출 성능)
- **Classification Accuracy**: > 0.85 (변조 분류 정확도)
- **Regression MAE**: < 0.1 (파라미터 추정 오차)

---

## ⚠️ 문제 해결

### 문제 1: "CUDA out of memory"
**해결책**:
```python
# 배치 사이즈를 줄이려면 run_complete_simulation.py의 config를 수정
# 또는 런타임 재시작 후 다시 실행
```

### 문제 2: "ModuleNotFoundError"
**해결책**:
```python
# 패키지 재설치
!pip install --no-cache-dir torch torchvision numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml
```

### 문제 3: "브랜치를 찾을 수 없습니다"
**해결책**:
```python
# 최신 코드 가져오기
!git fetch --all
!git checkout claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc
!git pull
```

### 문제 4: 실행이 너무 느림
**해결책**:
- GPU 런타임으로 변경: `런타임` → `런타임 유형 변경` → `GPU`
- 또는 `run_complete_simulation.py`에서 `num_epochs`를 30으로 감소

---

## 📚 추가 리소스

- **프로젝트 문서**: [README.md](README.md)
- **로컬 환경 가이드**: [QUICKSTART.md](QUICKSTART.md)
- **프로젝트 상세**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## 💡 유용한 팁

1. **GPU 활성화**: 훈련 속도가 2-3배 빠릅니다
2. **중간 저장**: 체크포인트는 자동으로 저장됩니다
3. **재시작**: 중단되어도 `experiments/` 폴더의 체크포인트에서 재개 가능
4. **커스터마이징**: `run_complete_simulation.py`의 config 변수를 수정하여 실험 조건 변경

---

## 🎓 다음 단계

시뮬레이션 완료 후:

1. **결과 분석**: JSON 파일과 그래프를 통해 성능 분석
2. **하이퍼파라미터 튜닝**: 에포크 수, 학습률, 모델 크기 조정
3. **커스텀 신호**: `src/data/signal_generator.py`에 새 변조 타입 추가
4. **실제 데이터**: 저장된 모델로 실제 IQ 데이터 분석

---

## ✅ 실행 체크리스트

- [ ] Google Colab 계정 있음
- [ ] GPU 런타임 활성화 (권장)
- [ ] 코드 셀 복사 & 실행
- [ ] 결과 확인 (이미지 & JSON)
- [ ] 결과 다운로드 (선택)

---

**Happy Signal Processing! 📡**

시뮬레이션이 성공적으로 완료되기를 바랍니다!
