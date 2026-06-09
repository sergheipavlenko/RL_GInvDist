"""
ppo_core.py
===========
Ядро алгоритма Proximal Policy Optimization (PPO) для обучения
RL-агента выбора мономиального порядка.

Реализует:
  - PPOBuffer        — буфер хранения траекторий
  - compute_gae()    — Generalized Advantage Estimation
  - ppo_update()     — обновление политики по буферу
  - train_ppo()      — полный цикл обучения

Author: Pavlenko Sergey
Group:  НПИбд-02-23, RUDN University
Date:   2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """Один переход в буфере траекторий."""
    state:      np.ndarray
    action:     int
    reward:     float
    log_prob:   float
    value:      float
    done:       bool


@dataclass
class PPOBuffer:
    """
    Буфер хранения траекторий для алгоритма PPO.

    Накапливает переходы и вычисляет преимущества GAE
    перед передачей в ppo_update().
    """
    gamma:      float = 0.99
    lam:        float = 0.95   # λ для GAE
    capacity:   int   = 10_000

    _transitions: List[Transition] = field(default_factory=list, repr=False)

    def store(self, state: np.ndarray, action: int, reward: float,
              log_prob: float, value: float, done: bool) -> None:
        """Добавить переход в буфер."""
        if len(self._transitions) >= self.capacity:
            self._transitions.pop(0)
        self._transitions.append(
            Transition(state, action, reward, log_prob, value, done)
        )

    def clear(self) -> None:
        self._transitions.clear()

    def __len__(self) -> int:
        return len(self._transitions)

    def get_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Вернуть словарь тензоров с GAE-преимуществами и returns.

        Returns:
            {states, actions, old_log_probs, advantages, returns}
        """
        n = len(self._transitions)
        states      = np.stack([t.state    for t in self._transitions])
        actions     = np.array([t.action   for t in self._transitions], dtype=np.int64)
        rewards     = np.array([t.reward   for t in self._transitions], dtype=np.float32)
        log_probs   = np.array([t.log_prob for t in self._transitions], dtype=np.float32)
        values      = np.array([t.value    for t in self._transitions], dtype=np.float32)
        dones       = np.array([t.done     for t in self._transitions], dtype=np.float32)

        advantages = compute_gae(rewards, values, dones, self.gamma, self.lam)
        returns    = advantages + values

        # Нормализация преимуществ
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states":       torch.FloatTensor(states).to(device),
            "actions":      torch.LongTensor(actions).to(device),
            "old_log_probs":torch.FloatTensor(log_probs).to(device),
            "advantages":   torch.FloatTensor(advantages).to(device),
            "returns":      torch.FloatTensor(returns).to(device),
        }


# ─────────────────────────────────────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: np.ndarray,
    values:  np.ndarray,
    dones:   np.ndarray,
    gamma:   float = 0.99,
    lam:     float = 0.95,
    last_value: float = 0.0,
) -> np.ndarray:
    """
    Generalized Advantage Estimation (Schulman et al., 2016).

    Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l},   δ_t = r_t + γ V(s_{t+1}) − V(s_t)

    Args:
        rewards:    массив наград r_t,  shape (T,)
        values:     оценки V(s_t),      shape (T,)
        dones:      флаги done (1.0 = конец эпизода), shape (T,)
        gamma:      коэффициент дисконтирования
        lam:        параметр λ для GAE
        last_value: оценка V(s_T) — используется для bootstrap

    Returns:
        advantages: shape (T,) dtype float32
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae

    return advantages


# ─────────────────────────────────────────────────────────────────────────────
# PPO update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    policy:       nn.Module,
    optimizer:    optim.Optimizer,
    buffer:       PPOBuffer,
    *,
    clip_epsilon: float = 0.2,
    value_coef:   float = 0.5,
    entropy_coef: float = 0.01,
    n_epochs:     int   = 4,
    batch_size:   int   = 64,
    max_grad_norm:float = 0.5,
    device:       torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Обновление политики PPO по накопленному буферу.

    Реализует Clipped Surrogate Loss:
        L_CLIP = E[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]
    с добавлением value loss и энтропийного бонуса.

    Args:
        policy:       нейронная сеть MonomorphicOrderPolicy
        optimizer:    оптимизатор Adam
        buffer:       PPOBuffer с накопленными переходами
        clip_epsilon: параметр клиппирования (ε = 0.2)
        value_coef:   коэффициент value loss
        entropy_coef: коэффициент энтропии
        n_epochs:     число проходов по буферу
        batch_size:   размер мини-батча
        max_grad_norm:порог градиентного клиппирования
        device:       устройство вычислений

    Returns:
        dict с метриками: policy_loss, value_loss, entropy, total_loss
    """
    if len(buffer) == 0:
        return {"policy_loss": 0., "value_loss": 0., "entropy": 0., "total_loss": 0.}

    data = buffer.get_tensors(device)
    states      = data["states"]
    actions     = data["actions"]
    old_log_prob= data["old_log_probs"]
    advantages  = data["advantages"]
    returns     = data["returns"]

    N = states.shape[0]
    metrics = {"policy_loss": 0., "value_loss": 0., "entropy": 0., "total_loss": 0.}

    for _ in range(n_epochs):
        # Случайное перемешивание
        indices = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            idx = indices[start: start + batch_size]

            s_b   = states[idx]
            a_b   = actions[idx]
            olp_b = old_log_prob[idx]
            adv_b = advantages[idx]
            ret_b = returns[idx]

            # Forward
            logits, values = policy(s_b)
            dist_b  = Categorical(logits=logits)
            nlp_b   = dist_b.log_prob(a_b)
            entropy = dist_b.entropy().mean()

            # Clipped surrogate loss
            ratio   = torch.exp(nlp_b - olp_b)
            surr1   = ratio * adv_b
            surr2   = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_b
            p_loss  = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            v_loss  = nn.functional.mse_loss(values.squeeze(-1), ret_b)

            # Total
            loss = p_loss + value_coef * v_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] += p_loss.item()
            metrics["value_loss"]  += v_loss.item()
            metrics["entropy"]     += entropy.item()
            metrics["total_loss"]  += loss.item()

    n_batches = n_epochs * max(1, N // batch_size)
    for k in metrics:
        metrics[k] /= n_batches

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    policy,
    env,
    optimizer:    optim.Optimizer,
    *,
    num_episodes: int   = 100,
    update_every: int   = 10,
    gamma:        float = 0.99,
    lam:          float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef:   float = 0.5,
    entropy_coef: float = 0.01,
    n_epochs:     int   = 4,
    batch_size:   int   = 64,
    max_grad_norm:float = 0.5,
    device:       torch.device = torch.device("cpu"),
    save_path:    Optional[str] = None,
) -> List[Dict]:
    """
    Полный цикл обучения PPO-агента.

    Args:
        policy:       MonomorphicOrderPolicy
        env:          RLEnvironment
        optimizer:    оптимизатор
        num_episodes: число эпизодов обучения
        update_every: обновлять политику каждые N эпизодов
        save_path:    путь для сохранения весов (опционально)
        прочие:       гиперпараметры PPO

    Returns:
        history: список словарей с метриками по эпизодам
    """
    buffer = PPOBuffer(gamma=gamma, lam=lam)
    history: List[Dict] = []
    episode_rewards: List[float] = []

    policy.to(device)
    policy.train()

    logger.info("PPO обучение запущено: %d эпизодов", num_episodes)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0

        while True:
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value = policy(s_t)
                dist_t   = Categorical(logits=logits)
                action_t = dist_t.sample()
                log_prob = dist_t.log_prob(action_t).item()

            action = action_t.item()
            next_state, reward, done = env.step(action)

            buffer.store(state, action, reward, log_prob, value.item(), done)
            ep_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(ep_reward)

        # Обновление политики
        if ep % update_every == 0 and len(buffer) >= batch_size:
            metrics = ppo_update(
                policy, optimizer, buffer,
                clip_epsilon=clip_epsilon, value_coef=value_coef,
                entropy_coef=entropy_coef, n_epochs=n_epochs,
                batch_size=batch_size, max_grad_norm=max_grad_norm,
                device=device,
            )
            buffer.clear()

            mean_r = float(np.mean(episode_rewards[-update_every:]))
            record = {"episode": ep, "mean_reward": mean_r, **metrics}
            history.append(record)

            logger.info(
                "Ep %4d/%d | reward=%.4f | p_loss=%.4f | v_loss=%.4f | entropy=%.4f",
                ep, num_episodes, mean_r,
                metrics["policy_loss"], metrics["value_loss"], metrics["entropy"],
            )

            # Сохранение лучшей модели
            if save_path and (
                not history[:-1] or mean_r >= max(h["mean_reward"] for h in history[:-1])
            ):
                torch.save(policy.state_dict(), save_path)
                logger.info("  ✓ Лучшая модель сохранена в %s", save_path)

    logger.info("Обучение завершено. Финальная средняя награда: %.4f",
                float(np.mean(episode_rewards[-update_every:])))
    return history


if __name__ == "__main__":
    # Быстрый smoke-test
    import sys
    sys.path.insert(0, ".")
    from rl_monomial_agent import MonomorphicOrderPolicy, RLEnvironment

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    policy    = MonomorphicOrderPolicy(state_dim=8, num_orders=3)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    env       = RLEnvironment(seed=42)

    history = train_ppo(policy, env, optimizer,
                        num_episodes=30, update_every=10,
                        save_path="models/best_model.pth")

    print("\n=== История обучения ===")
    for h in history:
        print(f"  ep={h['episode']:3d}  reward={h['mean_reward']:+.4f}"
              f"  p_loss={h['policy_loss']:.4f}  v_loss={h['value_loss']:.4f}")
