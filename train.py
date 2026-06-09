"""
train.py
========
Главный скрипт обучения и оценки RL-агента.

Использование:
    python train.py                       # обучение с параметрами по умолчанию
    python train.py --episodes 200        # задать число эпизодов
    python train.py --eval-only           # только оценка загруженной модели
    python train.py --config config.yaml  # использовать конфигурацию

Author: Pavlenko Sergey
Group:  НПИбд-02-23, RUDN University
Date:   2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from rl_monomial_agent import MonomorphicOrderPolicy, RLEnvironment
from ppo_core import train_ppo
from ginv_rl_wrapper import GInvRLWrapper
from feature_extractor import BENCHMARK_SYSTEMS

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RL-агент для выбора мономиального порядка (GInvDist)"
    )
    p.add_argument("--config",        default="config.yaml",
                   help="Путь к YAML-конфигурации")
    p.add_argument("--episodes",      type=int,   default=None,
                   help="Число эпизодов обучения (переопределяет config)")
    p.add_argument("--lr",            type=float, default=None,
                   help="Learning rate")
    p.add_argument("--device",        default="cpu",
                   help="Устройство: cpu или cuda")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--save-dir",      default="models",
                   help="Директория для сохранения модели")
    p.add_argument("--eval-only",     action="store_true",
                   help="Только оценка (загрузка модели из save-dir)")
    p.add_argument("--no-save",       action="store_true",
                   help="Не сохранять модель")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def run_training(cfg: dict, args: argparse.Namespace) -> list:
    t_cfg  = cfg.get("training", {})
    m_cfg  = cfg.get("model", {})

    num_episodes  = args.episodes  or t_cfg.get("num_episodes",  100)
    lr            = args.lr        or t_cfg.get("learning_rate", 3e-4)
    batch_size    = t_cfg.get("batch_size", 64)
    n_epochs      = t_cfg.get("n_epochs",   4)
    gamma         = t_cfg.get("gamma",      0.99)
    gae_lambda    = t_cfg.get("gae_lambda", 0.95)
    clip_ratio    = t_cfg.get("clip_ratio", 0.2)
    entropy_coef  = t_cfg.get("entropy_coeff", 0.01)
    value_coef    = t_cfg.get("value_loss_coeff", 0.5)
    grad_clip     = t_cfg.get("gradient_clip", 0.5)
    update_every  = t_cfg.get("update_every", 10)

    state_dim  = m_cfg.get("state_dim",   8)
    num_orders = m_cfg.get("num_actions", 3)
    hidden_dim = m_cfg.get("hidden_dim",  128)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("Запуск обучения PPO-агента")
    logger.info("  episodes=%d  lr=%.2e  gamma=%.3f  λ=%.3f",
                num_episodes, lr, gamma, gae_lambda)
    logger.info("  clip_ε=%.2f  batch=%d  epochs=%d  device=%s",
                clip_ratio, batch_size, n_epochs, args.device)
    logger.info("=" * 60)

    policy    = MonomorphicOrderPolicy(state_dim, num_orders, hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr,
                           weight_decay=t_cfg.get("weight_decay", 1e-5))
    env       = RLEnvironment(seed=args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    best_path = str(save_dir / "best_model.pth") if not args.no_save else None

    history = train_ppo(
        policy=policy,
        env=env,
        optimizer=optimizer,
        num_episodes=num_episodes,
        update_every=update_every,
        gamma=gamma,
        lam=gae_lambda,
        clip_epsilon=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        n_epochs=n_epochs,
        batch_size=batch_size,
        max_grad_norm=grad_clip,
        device=device,
        save_path=best_path,
    )

    # Сохранить финальную модель
    if not args.no_save:
        final_path = str(save_dir / "final_model.pth")
        torch.save(policy.state_dict(), final_path)
        logger.info("Финальная модель сохранена: %s", final_path)

        history_path = str(save_dir / "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                [{k: float(v) if isinstance(v, (float, int, np.floating)) else v
                  for k, v in h.items()}
                 for h in history],
                f, indent=2, ensure_ascii=False,
            )
        logger.info("История обучения сохранена: %s", history_path)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(save_dir: str, device: str = "cpu") -> None:
    """Загрузить обученную модель и оценить на всех бенчмарках."""
    model_path = Path(save_dir) / "best_model.pth"
    if not model_path.exists():
        model_path = Path(save_dir) / "final_model.pth"
    if not model_path.exists():
        logger.error("Модель не найдена в %s", save_dir)
        return

    wrapper = GInvRLWrapper.from_checkpoint(str(model_path), device=device)

    logger.info("\n%s", "=" * 65)
    logger.info("ОЦЕНКА АГЕНТА НА БЕНЧМАРКАХ")
    logger.info("%s", "=" * 65)
    logger.info("%-15s  %-12s  %8s  %8s  %s",
                "Система", "Порядок", "Время(с)", "Speedup", "Вероятности")
    logger.info("-" * 65)

    correct = 0
    total   = 0
    speedups = []

    # Оптимальный порядок для каждой системы (эталон)
    OPTIMAL = {s: "GREVLEX" for s in BENCHMARK_SYSTEMS}

    for sys_name in BENCHMARK_SYSTEMS:
        res    = wrapper.solve(system_name=sys_name)
        probs  = res["probabilities"]
        chosen = res["order_chosen"]
        is_ok  = (chosen == OPTIMAL.get(sys_name, "GREVLEX"))

        correct  += int(is_ok)
        total    += 1
        speedups.append(res["speedup_vs_lex"])

        logger.info(
            "%-15s  %-12s  %8.3f  %7.2fx  "
            "L=%.2f G=%.2f R=%.2f  %s",
            sys_name, chosen,
            res["time_sec"], res["speedup_vs_lex"],
            probs["LEX"], probs["GREVLEX"], probs["REVGREVLEX"],
            "✓" if is_ok else "✗",
        )

    logger.info("-" * 65)
    accuracy     = 100.0 * correct / max(total, 1)
    mean_speedup = float(np.mean(speedups))
    logger.info(
        "Точность: %d/%d = %.1f%%    Среднее ускорение: %.2fx",
        correct, total, accuracy, mean_speedup,
    )
    logger.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Загрузка конфигурации
    cfg = {}
    if Path(args.config).exists():
        cfg = load_config(args.config)
        logger.info("Конфигурация загружена из %s", args.config)
    else:
        logger.warning("Файл конфигурации %s не найден, используются параметры по умолчанию.", args.config)

    if args.eval_only:
        run_evaluation(args.save_dir, device=args.device)
    else:
        history = run_training(cfg, args)
        run_evaluation(args.save_dir, device=args.device)

        # Краткая сводка
        if history:
            final_reward = history[-1]["mean_reward"]
            print(f"\n✅ Обучение завершено. Финальная средняя награда: {final_reward:+.4f}")


if __name__ == "__main__":
    main()
