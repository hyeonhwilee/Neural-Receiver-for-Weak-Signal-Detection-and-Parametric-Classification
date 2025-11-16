#!/usr/bin/env python
"""
완전한 Neural Receiver 시뮬레이션 스크립트
Google Colab 및 로컬 환경에서 실행 가능
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

def print_banner(text):
    """배너 출력"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          Neural Receiver - Complete Simulation Pipeline         ║
    ║                                                                  ║
    ║     Weak Signal Detection and Parametric Classification         ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # ================================================================
    # 1. 환경 설정
    # ================================================================
    print_banner("1/8: 환경 설정")

    # 랜덤 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

    # 모듈 임포트
    from src.data import SignalGenerator, SignalDataset
    from src.models import NeuralReceiver
    from src.training import Trainer
    from src.evaluation import Evaluator

    print("✓ 모든 모듈 임포트 완료")

    # ================================================================
    # 2. 신호 생성 예제
    # ================================================================
    print_banner("2/8: 신호 생성 데모")

    signal_gen = SignalGenerator(sample_rate=1.0, seed=42)
    modulation_types = ['No-signal', 'AM', 'FM', '2FSK', 'CW Radar', 'LFM/Chirp']
    sequence_length = 1024
    snr_db = -5.0

    print(f"생성 중: {len(modulation_types)}개 신호 타입 (SNR = {snr_db} dB)")

    # SignalParams import
    from src.data.signal_generator import SignalParams

    signals = []
    for mod_type in modulation_types:
        # 파라미터 직접 생성
        # symbol_rate를 정규화된 값으로 설정 (0.01 ~ 0.2 = 각 심볼당 5~100 샘플)
        params = SignalParams(
            center_freq=np.random.uniform(-0.3, 0.3),
            bandwidth=np.random.uniform(0.05, 0.2),
            power=1.0 if mod_type != 'No-signal' else 0.0,
            snr_db=snr_db if mod_type != 'No-signal' else -np.inf,
            symbol_rate=np.random.uniform(0.01, 0.2),
            modulation_type=mod_type
        )
        signal, noisy_signal = signal_gen.generate_signal(sequence_length, params)
        signals.append((mod_type, noisy_signal, params))
        print(f"  ✓ {mod_type}")

    # 신호 시각화
    fig, axes = plt.subplots(len(modulation_types), 2, figsize=(15, 3*len(modulation_types)))

    for idx, (mod_type, signal, params) in enumerate(signals):
        # Time domain
        ax1 = axes[idx, 0]
        time = np.arange(len(signal))
        ax1.plot(time[:200], signal.real[:200], label='I', alpha=0.7)
        ax1.plot(time[:200], signal.imag[:200], label='Q', alpha=0.7)
        ax1.set_title(f'{mod_type} - Time Domain (SNR={snr_db} dB)')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Frequency domain
        ax2 = axes[idx, 1]
        fft = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal)))
        ax2.plot(freq, 20*np.log10(np.abs(fft) + 1e-10))
        ax2.set_title(f'{mod_type} - Frequency Domain')
        ax2.set_xlabel('Normalized Frequency')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(params.center_freq, color='r', linestyle='--', alpha=0.5, label='Center')
        ax2.legend()

    plt.tight_layout()
    plt.savefig('signal_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 신호 예제 저장: signal_examples.png")

    # ================================================================
    # 3. 데이터셋 준비
    # ================================================================
    print_banner("3/8: 데이터셋 준비")

    config = {
        'sequence_length': 1024,
        'snr_range': (-10, 0),
        'train_samples': 10000,
        'val_samples': 2000,
        'test_samples': 2000,
        'batch_size': 64,
        'num_workers': 2
    }

    print(f"훈련 샘플: {config['train_samples']:,}")
    print(f"검증 샘플: {config['val_samples']:,}")
    print(f"테스트 샘플: {config['test_samples']:,}")
    print(f"SNR 범위: {config['snr_range'][0]} ~ {config['snr_range'][1]} dB")

    train_dataset = SignalDataset(
        num_samples=config['train_samples'],
        sequence_length=config['sequence_length'],
        snr_range=config['snr_range'],
        seed=42
    )

    val_dataset = SignalDataset(
        num_samples=config['val_samples'],
        sequence_length=config['sequence_length'],
        snr_range=config['snr_range'],
        seed=43
    )

    test_dataset = SignalDataset(
        num_samples=config['test_samples'],
        sequence_length=config['sequence_length'],
        snr_range=config['snr_range'],
        seed=44
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])

    print(f"✓ 데이터 로더 생성 완료 (batch_size={config['batch_size']})")

    # ================================================================
    # 4. 모델 생성
    # ================================================================
    print_banner("4/8: 모델 생성")

    model_config = {
        'input_channels': 2,
        'base_channels': 64,
        'num_blocks': 4,
        'num_classes': len(train_dataset.modulation_classes),
        'num_params': 4,
        'dropout': 0.3
    }

    model = NeuralReceiver(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ 모델 생성 완료")
    print(f"  총 파라미터: {total_params:,}")
    print(f"  훈련 가능 파라미터: {trainable_params:,}")
    print(f"  변조 클래스 수: {model_config['num_classes']}")

    # ================================================================
    # 5. 훈련
    # ================================================================
    print_banner("5/8: 모델 훈련")

    training_config = {
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'save_dir': 'experiments/neural_receiver',
        'loss_weights': {
            'detection': 1.0,
            'classification': 1.0,
            'regression': 1.0
        }
    }

    print(f"에포크 수: {training_config['num_epochs']}")
    print(f"학습률: {training_config['learning_rate']}")
    print(f"예상 소요 시간: {'5-10분 (GPU)' if torch.cuda.is_available() else '15-20분 (CPU)'}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        save_dir=training_config['save_dir'],
        loss_weights=training_config['loss_weights']
    )

    print("\n훈련 시작...\n")
    history = trainer.train(num_epochs=training_config['num_epochs'])
    print("\n✓ 훈련 완료!")

    # 훈련 히스토리 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_detection_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_detection_acc'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Detection Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['train_classification_acc'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_classification_acc'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Modulation Classification Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['train_regression_mae'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_regression_mae'], label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Parameter Estimation MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 훈련 히스토리 저장: training_history.png")

    # ================================================================
    # 6. 모델 로드
    # ================================================================
    print_banner("6/8: 최적 모델 로드")

    best_model_path = Path(training_config['save_dir']) / 'checkpoints' / 'best_model.pt'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 최적 모델 로드 완료 (epoch {checkpoint['epoch']})")
    print(f"  검증 손실: {checkpoint['val_loss']:.4f}")

    # ================================================================
    # 7. 평가
    # ================================================================
    print_banner("7/8: 모델 평가")

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=train_dataset.modulation_classes
    )

    print("평가 실행 중...")
    results = evaluator.evaluate()

    print("\n" + "="*70)
    print("📊 검출 메트릭")
    print("="*70)
    print(f"ROC AUC:        {results['detection']['roc_auc']:.4f}")
    print(f"Accuracy:       {results['detection']['accuracy']:.4f}")
    print(f"Precision:      {results['detection']['precision']:.4f}")
    print(f"Recall:         {results['detection']['recall']:.4f}")
    print(f"F1 Score:       {results['detection']['f1']:.4f}")

    print("\n" + "="*70)
    print("📊 분류 메트릭")
    print("="*70)
    print(f"Accuracy:       {results['classification']['accuracy']:.4f}")
    print(f"Macro F1:       {results['classification']['f1_macro']:.4f}")
    print(f"Weighted F1:    {results['classification']['f1_weighted']:.4f}")

    print("\n" + "="*70)
    print("📊 회귀 메트릭")
    print("="*70)
    print(f"Overall MAE:    {results['regression']['mae']:.4f}")
    print(f"Overall RMSE:   {results['regression']['rmse']:.4f}")
    print(f"\nPer-parameter MAE:")
    param_names = ['Center Freq', 'Bandwidth', 'SNR', 'Symbol Rate']
    for name, mae in zip(param_names, results['regression']['mae_per_param']):
        print(f"  {name:15s}: {mae:.4f}")
    print("="*70)

    # ================================================================
    # 8. 결과 시각화
    # ================================================================
    print_banner("8/8: 결과 시각화")

    # ROC 곡선
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(results['detection']['fpr'], results['detection']['tpr'],
                 linewidth=2, label=f"ROC (AUC = {results['detection']['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve - Signal Detection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['detection']['recall_curve'], results['detection']['precision_curve'],
                 linewidth=2, label='PR Curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('detection_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 검출 메트릭 저장: detection_metrics.png")

    # 혼동 행렬
    fig, ax = plt.subplots(figsize=(12, 10))
    cm = results['classification']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.modulation_classes,
                yticklabels=train_dataset.modulation_classes,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - Modulation Classification')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 혼동 행렬 저장: confusion_matrix.png")

    # 회귀 오차
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    predictions = results['regression']['predictions']
    targets = results['regression']['targets']

    for idx, (ax, param_name) in enumerate(zip(axes.flatten(), param_names)):
        pred = predictions[:, idx]
        true = targets[:, idx]

        ax.scatter(true, pred, alpha=0.3, s=10)

        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true)**2))

        ax.set_xlabel(f'True {param_name}')
        ax.set_ylabel(f'Predicted {param_name}')
        ax.set_title(f'{param_name}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('regression_errors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 회귀 오차 저장: regression_errors.png")

    # 결과 저장
    final_results = {
        'config': {
            'dataset': config,
            'model': model_config,
            'training': training_config
        },
        'metrics': {
            'detection': {
                'roc_auc': float(results['detection']['roc_auc']),
                'accuracy': float(results['detection']['accuracy']),
                'f1': float(results['detection']['f1'])
            },
            'classification': {
                'accuracy': float(results['classification']['accuracy']),
                'f1_macro': float(results['classification']['f1_macro']),
                'f1_weighted': float(results['classification']['f1_weighted'])
            },
            'regression': {
                'mae': float(results['regression']['mae']),
                'rmse': float(results['regression']['rmse'])
            }
        }
    }

    with open('simulation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print("✓ 결과 저장: simulation_results.json")

    # 최종 요약
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 20 + "시뮬레이션 완료!" + " " * 28 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    print(f"\n{'생성된 파일':<30s}")
    print("-" * 70)
    for file in ['signal_examples.png', 'training_history.png',
                 'detection_metrics.png', 'confusion_matrix.png',
                 'regression_errors.png', 'simulation_results.json']:
        print(f"  ✓ {file}")

    print(f"\n{'최종 성능':<30s}")
    print("-" * 70)
    print(f"  Detection ROC-AUC:          {results['detection']['roc_auc']:.4f}")
    print(f"  Classification Accuracy:    {results['classification']['accuracy']:.4f}")
    print(f"  Parameter MAE:              {results['regression']['mae']:.4f}")

    print("\n" + "#" * 70)
    print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")
    print("="*70 + "\n")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
