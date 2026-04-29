#!/usr/bin/env bash
set -euo pipefail

echo "==> Detecting Ubuntu version"
if ! command -v lsb_release >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y lsb-release wget gnupg
fi

UBUNTU_VERSION="$(lsb_release -rs)"
case "$UBUNTU_VERSION" in
  20.04) CUDA_REPO="ubuntu2004" ;;
  22.04) CUDA_REPO="ubuntu2204" ;;
  24.04) CUDA_REPO="ubuntu2404" ;;
  *)
    echo "Unsupported Ubuntu version: $UBUNTU_VERSION"
    echo "This script supports Ubuntu 20.04, 22.04, or 24.04 in WSL."
    exit 1
    ;;
esac

echo "==> Installing prerequisites"
sudo apt-get update
sudo apt-get install -y wget gnupg software-properties-common

echo "==> Installing NVIDIA CUDA repo keyring"
wget -qO /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i /tmp/cuda-keyring.deb

echo "==> Updating apt metadata"
sudo apt-get update

echo "==> Installing CUDA Toolkit 12.4 only"
# Toolkit only. In WSL, the GPU driver belongs on Windows, not inside WSL.
sudo apt-get install -y cuda-toolkit-12-4

echo "==> Pointing /usr/local/cuda to 12.4"
sudo ln -sfn /usr/local/cuda-12.4 /usr/local/cuda

echo "==> Cleaning old CUDA env lines from ~/.bashrc"
cp ~/.bashrc ~/.bashrc.backup.$(date +%s)
sed -i '/cuda-11\.8/d' ~/.bashrc
sed -i '/cuda-12\.[0-9]/d' ~/.bashrc
sed -i '/\/usr\/local\/cuda\/bin/d' ~/.bashrc
sed -i '/CUDA_HOME/d' ~/.bashrc
sed -i '/LD_LIBRARY_PATH=.*cuda/d' ~/.bashrc
sed -i '/PATH=.*cuda/d' ~/.bashrc

echo "==> Writing fresh CUDA 12.4 env to ~/.bashrc"
cat >> ~/.bashrc <<'EOF'

# CUDA 12.4
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
EOF

echo "==> Exporting CUDA 12.4 into current shell"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

echo "==> Verifying toolkit detection"
which nvcc || true
nvcc --version

echo "==> Verifying /usr/local/cuda symlink"
ls -l /usr/local/cuda

echo "==> Verifying PyTorch sees the expected toolkit path"
python - <<'PY'
import os
print("CUDA_HOME =", os.environ.get("CUDA_HOME"))
try:
    import torch
    print("torch.__version__ =", torch.__version__)
    print("torch.version.cuda =", torch.version.cuda)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
except Exception as e:
    print("PyTorch import failed:", e)
PY

echo
echo "Done."
echo "Open a NEW shell or run: source ~/.bashrc"
echo "Important: nvidia-smi may show a driver CUDA version; for builds, nvcc/CUDA_HOME are the key checks."