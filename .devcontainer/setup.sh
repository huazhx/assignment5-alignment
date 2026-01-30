#!/bin/bash
set -e

echo "=== Installing dependencies as $(whoami) ==="

# 1. 安装基础工具 (使用了 sudo，因为是普通用户)
# 注意：Dockerfile 里已经给了 sudo 权限
sudo apt-get update && sudo apt-get install -y git ninja-build wget

# 2. 安装 uv (安装到 ~/.local/bin，已经在 Dockerfile 加到 PATH 了)
wget -qO- https://astral.sh/uv/install.sh | sh

# 3. 初始化环境
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

source .venv/bin/activate

echo "Installing project dependencies..."
if [ -f "pyproject.toml" ] || [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt || uv sync
fi

echo "=== Setup Complete ==="