#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=== 课程笔记检索助手 ==="
echo ""

# 1. Install dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt --quiet 2>/dev/null || pip install -r requirements.txt

# 2. Create data dirs
mkdir -p uploads db

# 3. Start Flask
echo "[2/3] Starting server on http://0.0.0.0:5000 ..."
echo "[3/3] Open the URL in your browser."
echo ""
python app.py