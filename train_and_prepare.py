import subprocess
import sys

def run(cmd):
    print(f"\nRunning: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")

if __name__ == "__main__":
    run("python -m app.training.train_resnet34")
    run("python -m app.support.save_support_embeddings")
    print("\nTraining + support embedding preparation complete.")
