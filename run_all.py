"""데이터만 넣으면 전부 자동: 종류 판별 → 알맞은 파이프라인 → 제출.

    python run_all.py
    python run_all.py --task tabular --tabular_model lightgbm
    python run_all.py --deep_model resnet34 --feature mel
"""
import argparse
from src.config import load_config
from src.pipeline import run_auto

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--task", default=None)
    ap.add_argument("--deep_model", default=None)
    ap.add_argument("--tabular_model", default=None)
    ap.add_argument("--feature", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, overrides={
        "task.type": a.task, "model.deep_name": a.deep_model,
        "model.tabular_name": a.tabular_model, "audio.feature": a.feature})
    run_auto(cfg)
