#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m streamlit run ui/dashboard.py
