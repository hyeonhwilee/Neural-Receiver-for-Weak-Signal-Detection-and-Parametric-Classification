"""
Google Colab에서 Neural Receiver 시뮬레이션을 실행하는 스크립트

이 스크립트는 다음 작업을 수행합니다:
1. 저장소 클론
2. 올바른 브랜치로 체크아웃
3. 필요한 패키지 설치
4. run_simulation.ipynb 노트북 실행
5. 결과 표시
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*70}")
    print(f"📌 {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"⚠️ Error: {result.stderr}", file=sys.stderr)
    return result.returncode

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        Neural Receiver Simulation - Google Colab Runner         ║
    ║                                                                  ║
    ║     Weak Signal Detection and Parametric Classification         ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # 1. 저장소 클론
    repo_url = "https://github.com/hyeonhwilee/Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification.git"
    repo_dir = "Neural-Receiver-for-Weak-Signal-Detection-and-Parametric-Classification"

    if not os.path.exists(repo_dir):
        run_command(
            f"git clone {repo_url}",
            "1/5: 저장소 클론 중..."
        )
    else:
        print(f"\n✓ 저장소가 이미 존재합니다: {repo_dir}")

    # 2. 디렉토리 이동
    os.chdir(repo_dir)
    print(f"\n✓ 작업 디렉토리: {os.getcwd()}")

    # 3. 올바른 브랜치로 체크아웃
    run_command(
        "git checkout claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc",
        "2/5: 브랜치 체크아웃 중..."
    )

    # 4. 최신 코드 pull
    run_command(
        "git pull origin claude/neural-receiver-simulation-01HfE6hbQ8oCoSBiPsAvpruc",
        "3/5: 최신 코드 가져오기..."
    )

    # 5. 필요한 패키지 설치
    run_command(
        "pip install -q torch torchvision tensorboard numpy scipy scikit-learn matplotlib seaborn tqdm pyyaml",
        "4/5: 필요한 패키지 설치 중... (1-2분 소요)"
    )

    # 6. nbconvert 설치
    run_command(
        "pip install -q nbformat nbconvert ipython",
        "설치: Jupyter 노트북 실행 도구"
    )

    print(f"\n{'='*70}")
    print("5/5: run_simulation.ipynb 노트북 실행 중...")
    print("⏱️  예상 소요 시간: 10-15분 (GPU 사용 시 5-10분)")
    print(f"{'='*70}\n")

    # 7. 노트북을 Python 스크립트로 변환하여 실행
    if os.path.exists("run_simulation.ipynb"):
        # nbconvert로 스크립트 변환
        subprocess.run(
            "jupyter nbconvert --to script run_simulation.ipynb --output run_simulation_exec",
            shell=True
        )

        # Python 스크립트 실행
        result = subprocess.run(
            "python run_simulation_exec.py",
            shell=True,
            capture_output=False
        )

        if result.returncode == 0:
            print(f"\n{'='*70}")
            print("✅ 시뮬레이션 완료!")
            print(f"{'='*70}\n")

            # 결과 파일 확인
            print("📊 생성된 결과 파일:")
            result_files = [
                "simulation_results.json",
                "signal_examples.png",
                "training_history.png",
                "detection_metrics.png",
                "confusion_matrix.png",
                "regression_errors.png"
            ]

            for file in result_files:
                if os.path.exists(file):
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ {file} (생성되지 않음)")

            # 모델 체크포인트 확인
            checkpoint_dir = Path("experiments/neural_receiver/checkpoints")
            if checkpoint_dir.exists():
                print(f"\n📁 모델 체크포인트: {checkpoint_dir}")
                for ckpt in checkpoint_dir.glob("*.pt"):
                    print(f"  ✓ {ckpt.name}")
        else:
            print(f"\n❌ 시뮬레이션 실행 중 오류 발생")
            return 1
    else:
        print("❌ run_simulation.ipynb 파일을 찾을 수 없습니다!")
        return 1

    print(f"\n{'='*70}")
    print("🎉 모든 작업이 완료되었습니다!")
    print(f"{'='*70}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
