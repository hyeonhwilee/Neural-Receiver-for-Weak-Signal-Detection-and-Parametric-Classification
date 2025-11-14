#!/usr/bin/env python3
"""
Simple runner script for SIGINT simulation
"""

import sys
import subprocess

def main():
    print("Starting SIGINT Neural Receiver Simulation...")
    print("-" * 70)

    try:
        # Run the simulation
        result = subprocess.run(
            [sys.executable, "sigint_simulation.py"],
            check=True,
            capture_output=False
        )

        print("\n" + "=" * 70)
        print("Simulation completed successfully!")
        print("=" * 70)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Simulation failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Error: sigint_simulation.py not found!")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        sys.exit(130)

if __name__ == "__main__":
    main()
