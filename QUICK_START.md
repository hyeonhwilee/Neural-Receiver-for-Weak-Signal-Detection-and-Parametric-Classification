# ⚡ Colab 원클릭 실행 가이드

## 🎯 가장 빠른 방법: 전체 시뮬레이션 노트북

### 1. Colab에서 바로 열기
아래 링크를 클릭하면 전체 시뮬레이션이 준비된 노트북이 열립니다:

```
https://colab.research.google.com/github/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification/blob/main/run_simulation.ipynb
```

**실행 방법:**
1. 위 링크 클릭
2. 런타임 > 런타임 유형 변경 > GPU 선택
3. 런타임 > 모두 실행 (Ctrl+F9)
4. 결과 확인! ✅

---

## 📝 새 노트북에서 코드 복사 실행

Google Colab에서 새 노트북을 만들고 아래 코드를 **하나의 셀에** 복사하여 실행하세요.

### 원클릭 실행 코드

```python
# ==========================================
# Neural Receiver 전체 시뮬레이션 원클릭 실행
# ==========================================

# Step 1: 저장소 클론 및 환경 설정
!git clone https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git
%cd Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification

# Step 2: 패키지 설치
!pip install -q torch torchvision torchaudio numpy scipy matplotlib seaborn scikit-learn pandas tqdm

# Step 3: GPU 확인
import torch
print(f"\n{'='*60}")
print(f"🔧 환경 설정 완료")
print(f"{'='*60}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"{'='*60}\n")

# Step 4: 간단한 테스트 실행
print("✅ 모든 준비가 완료되었습니다!")
print("📊 이제 run_simulation.ipynb를 열어서 전체 시뮬레이션을 실행하거나,")
print("💻 아래 코드를 실행하여 간단한 예제를 확인하세요.\n")
```

---

## 🎨 간단한 데모 코드

환경 설정 후 아래 코드를 실행하면 IQ 신호 예제를 바로 확인할 수 있습니다:

```python
import numpy as np
import matplotlib.pyplot as plt

# IQ 신호 생성 함수
def generate_iq_demo():
    # 신호 파라미터
    num_samples = 1000
    frequency = 5
    snr_db = 0

    # 시간 벡터
    t = np.linspace(0, 1, num_samples)

    # 신호 생성
    signal = np.exp(1j * 2 * np.pi * frequency * t)

    # SNR 적용
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.sqrt(snr_linear)
    signal = signal * signal_power

    # 노이즈 추가
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.1
    iq_signal = signal + noise

    return iq_signal

# 신호 생성
iq = generate_iq_demo()

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# I 성분
axes[0].plot(iq.real[:200])
axes[0].set_title('I (In-phase) Component')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True)

# Q 성분
axes[1].plot(iq.imag[:200])
axes[1].set_title('Q (Quadrature) Component')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True)

# IQ 평면 (Constellation)
axes[2].scatter(iq.real, iq.imag, alpha=0.3, s=5)
axes[2].set_title('IQ Constellation')
axes[2].set_xlabel('I')
axes[2].set_ylabel('Q')
axes[2].grid(True)
axes[2].axis('equal')

plt.tight_layout()
plt.show()

print("✅ IQ 신호 시각화 완료!")
print(f"📊 생성된 샘플 수: {len(iq)}")
```

---

## 🔥 전체 시뮬레이션 실행 (고급)

더 자세한 전체 시뮬레이션을 실행하려면:

1. **run_simulation.ipynb** 파일을 Colab에서 엽니다
2. 또는 아래 단계를 따라 직접 코드를 실행합니다:

```python
# run_simulation.ipynb의 모든 코드를 순차적으로 실행하거나
# 아래 링크에서 노트북을 열어 실행하세요
```

**링크:** [run_simulation.ipynb 열기](https://colab.research.google.com/github/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification/blob/main/run_simulation.ipynb)

---

## 📋 실행 체크리스트

- [ ] GPU 활성화 (런타임 > 런타임 유형 변경 > GPU)
- [ ] 저장소 클론 완료
- [ ] 패키지 설치 완료
- [ ] CUDA 사용 가능 확인
- [ ] 예제 코드 실행 성공
- [ ] 전체 시뮬레이션 실행

---

## 💡 팁

1. **GPU 사용**: GPU를 사용하면 학습 속도가 10배 이상 빨라집니다
   - 런타임 > 런타임 유형 변경 > 하드웨어 가속기 > GPU

2. **세션 유지**: Colab 무료 버전은 12시간 후 세션이 종료됩니다
   - 중요한 결과는 Google Drive에 저장하세요

3. **메모리 부족**: 메모리 오류 발생 시
   - 런타임 > 런타임 다시 시작
   - batch_size를 줄여보세요 (64 → 32)

---

## 🎯 예상 결과

전체 시뮬레이션을 실행하면 다음 결과를 얻을 수 있습니다:

- ✅ 신호 검출 정확도: **~90-95%**
- ✅ 변조 분류 정확도: **~85-90%** (높은 SNR에서)
- ✅ ROC-AUC: **~0.95+**
- ✅ SNR별 성능 그래프
- ✅ Confusion Matrix
- ✅ 학습 곡선 시각화

실행 시간: **약 5-10분** (GPU 사용 시)

---

## 🆘 문제 해결

**문제**: 패키지 설치 오류
- **해결**: `!pip install --upgrade pip` 실행 후 재시도

**문제**: CUDA 사용 불가
- **해결**: 런타임 > 런타임 유형 변경 > GPU 선택 확인

**문제**: 메모리 부족
- **해결**: 런타임 다시 시작, batch_size 줄이기

**문제**: 저장소 클론 실패
- **해결**: 이미 클론되어 있을 수 있습니다. `%cd` 명령어로 디렉토리 이동

---

**질문이나 이슈가 있으면 GitHub Issues에 등록해주세요!** 🙏
