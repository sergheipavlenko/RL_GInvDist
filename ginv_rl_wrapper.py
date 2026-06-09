"""
ginv_rl_wrapper.py
==================
Модуль интеграции RL-агента с библиотекой GInvDist v2.1.0.

Предоставляет класс GInvRLWrapper, который:
  1. Использует обученный PPO-агент для предсказания оптимального
     мономиального порядка по структурным признакам системы.
  2. Передаёт выбранный порядок в GInvDist через метод
     set_monomial_order_by_permutation().
  3. Запускает вычисление базиса Грёбнера и возвращает результат.

Если GInvDist не установлен — используется режим симуляции (simulation_mode=True),
опирающийся на предвычисленные данные из RLEnvironment.

Author: Pavlenko Sergey
Group:  НПИбд-02-23, RUDN University
Date:   2026
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from feature_extractor import PolynomialSystemFeatureExtractor, BENCHMARK_SYSTEMS
from rl_monomial_agent import MonomorphicOrderPolicy, RLEnvironment

logger = logging.getLogger(__name__)

# Попытка импорта реальной GInvDist
try:
    import ginv  # type: ignore
    GINVDIST_AVAILABLE = True
    logger.info("GInvDist успешно импортирован.")
except ImportError:
    GINVDIST_AVAILABLE = False
    logger.warning(
        "GInvDist не найден. GInvRLWrapper будет работать в режиме симуляции."
    )

# Словарь: индекс действия → имя порядка
ACTION_TO_ORDER = {0: "LEX", 1: "GREVLEX", 2: "REVGREVLEX"}
ORDER_TO_INDEX  = {v: k for k, v in ACTION_TO_ORDER.items()}

# Словарь: имя порядка → перестановка переменных для GInvDist API
# (перестановка определяет лексикографический вес переменных)
ORDER_TO_PERMUTATION = {
    "LEX":        lambda n: list(range(n)),               # x0 > x1 > ... > x_{n-1}
    "GREVLEX":    lambda n: list(range(n - 1, -1, -1)),   # x_{n-1} > ... > x0
    "REVGREVLEX": lambda n: list(range(n - 1, -1, -1)),   # аналогично, флаг reversed
}


class GInvRLWrapper:
    """
    Обёртка для интеграции RL-агента с GInvDist.

    Пример использования::

        wrapper = GInvRLWrapper.from_checkpoint("models/best_model.pth")

        # Автоматический выбор порядка и вычисление базиса
        result = wrapper.solve(
            polynomials=my_polys,
            variables=my_vars,
        )
        print(result["order_chosen"], result["basis"], result["time_sec"])

    При отсутствии GInvDist работает в режиме симуляции,
    возвращая данные из BENCHMARK_SYSTEMS.
    """

    def __init__(
        self,
        policy:          MonomorphicOrderPolicy,
        simulation_mode: bool = not GINVDIST_AVAILABLE,
        device:          str  = "cpu",
    ):
        self.policy          = policy
        self.simulation_mode = simulation_mode
        self.device          = torch.device(device)
        self.extractor       = PolynomialSystemFeatureExtractor()
        self._env_sim        = RLEnvironment(seed=0) if simulation_mode else None

        self.policy.to(self.device)
        self.policy.eval()

        mode_str = "SIMULATION" if simulation_mode else "REAL GInvDist"
        logger.info("GInvRLWrapper инициализирован (режим: %s)", mode_str)

    # ─────────────────────────────────────────────────────────────────
    # Factory methods
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        state_dim:       int = 8,
        num_orders:      int = 3,
        device:          str = "cpu",
    ) -> "GInvRLWrapper":
        """Загрузить обёртку из сохранённых весов."""
        policy = MonomorphicOrderPolicy(state_dim=state_dim, num_orders=num_orders)
        state_dict = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(state_dict)
        logger.info("Веса загружены из %s", checkpoint_path)
        return cls(policy=policy, device=device)

    @classmethod
    def with_default_policy(cls, device: str = "cpu") -> "GInvRLWrapper":
        """Создать обёртку с необученной (случайной) политикой — для тестирования."""
        policy = MonomorphicOrderPolicy()
        return cls(policy=policy, device=device)

    # ─────────────────────────────────────────────────────────────────
    # Core public API
    # ─────────────────────────────────────────────────────────────────

    def predict_order(
        self,
        system_dict: Optional[Dict] = None,
        polynomials=None,
        variables=None,
    ) -> Tuple[int, str, np.ndarray]:
        """
        Предсказать оптимальный мономиальный порядок.

        Args:
            system_dict: словарь с параметрами системы (см. PolynomialSystemFeatureExtractor)
            polynomials: список SymPy-полиномов (альтернатива system_dict)
            variables:   список SymPy-переменных

        Returns:
            (action_idx, order_name, features)
              action_idx:  0=LEX, 1=GREVLEX, 2=REVGREVLEX
              order_name:  строковое имя порядка
              features:    вектор признаков shape (8,)
        """
        features = self.extractor.extract(
            polynomials=polynomials,
            variables=variables,
            system_dict=system_dict,
        )

        state = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.policy(state)
            action = int(torch.argmax(logits, dim=-1).item())

        order_name = ACTION_TO_ORDER[action]
        logger.debug("Предсказан порядок: %s (action=%d)", order_name, action)
        return action, order_name, features

    def predict_order_proba(
        self,
        system_dict: Optional[Dict] = None,
        polynomials=None,
        variables=None,
    ) -> Dict[str, float]:
        """
        Вернуть вероятности каждого порядка.

        Returns:
            {"LEX": p0, "GREVLEX": p1, "REVGREVLEX": p2}
        """
        features = self.extractor.extract(
            polynomials=polynomials,
            variables=variables,
            system_dict=system_dict,
        )
        state = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.policy(state)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        return {ACTION_TO_ORDER[i]: float(probs[i]) for i in range(3)}

    def solve(
        self,
        polynomials=None,
        variables=None,
        system_dict: Optional[Dict]  = None,
        system_name: Optional[str]   = None,
        force_order: Optional[str]   = None,
    ) -> Dict[str, Any]:
        """
        Автоматически выбрать порядок и вычислить базис Грёбнера.

        Args:
            polynomials: список SymPy-полиномов (для реального режима)
            variables:   список SymPy-переменных
            system_dict: предвычисленные параметры системы
            system_name: имя бенчмарка (для симуляции)
            force_order: принудительно задать порядок ("LEX"/"GREVLEX"/"REVGREVLEX")

        Returns:
            dict с ключами:
              order_chosen   — выбранный порядок
              action_idx     — индекс действия (0/1/2)
              probabilities  — вероятности порядков
              time_sec       — время вычисления (с)
              basis          — базис Грёбнера (список или заглушка)
              speedup_vs_lex — оцененное ускорение vs LEX
              features       — вектор признаков
              simulation     — True если режим симуляции
        """
        # 1. Определить вектор признаков
        if system_name and system_name in BENCHMARK_SYSTEMS:
            system_dict = system_dict or BENCHMARK_SYSTEMS[system_name]

        # 2. Предсказание порядка
        if force_order:
            order_name = force_order.upper()
            action_idx = ORDER_TO_INDEX.get(order_name, 1)
            features   = self.extractor.extract(
                polynomials=polynomials,
                variables=variables,
                system_dict=system_dict,
            )
        else:
            action_idx, order_name, features = self.predict_order(
                system_dict=system_dict,
                polynomials=polynomials,
                variables=variables,
            )

        probabilities = self.predict_order_proba(
            system_dict=system_dict,
            polynomials=polynomials,
            variables=variables,
        )

        # 3. Вычисление базиса
        if self.simulation_mode or not GINVDIST_AVAILABLE:
            result = self._simulate_computation(
                system_name=system_name,
                order_name=order_name,
                features=features,
            )
        else:
            result = self._real_computation(
                polynomials=polynomials,
                variables=variables,
                order_name=order_name,
            )

        result.update({
            "order_chosen":  order_name,
            "action_idx":    action_idx,
            "probabilities": probabilities,
            "features":      features.tolist(),
            "simulation":    self.simulation_mode or not GINVDIST_AVAILABLE,
        })

        logger.info(
            "solve() → порядок=%s  время=%.4f с  ускорение=%.2fx",
            order_name, result["time_sec"], result.get("speedup_vs_lex", 1.0),
        )
        return result

    def benchmark_all_orders(
        self,
        system_name: Optional[str]  = None,
        system_dict: Optional[Dict] = None,
        polynomials=None,
        variables=None,
    ) -> Dict[str, Dict]:
        """
        Прогнать систему со всеми тремя порядками и сравнить результаты.

        Returns:
            {"LEX": {...}, "GREVLEX": {...}, "REVGREVLEX": {...}}
        """
        results = {}
        for order in ["LEX", "GREVLEX", "REVGREVLEX"]:
            results[order] = self.solve(
                polynomials=polynomials,
                variables=variables,
                system_dict=system_dict,
                system_name=system_name,
                force_order=order,
            )
        return results

    # ─────────────────────────────────────────────────────────────────
    # Internal: simulation
    # ─────────────────────────────────────────────────────────────────

    def _simulate_computation(
        self,
        system_name: Optional[str],
        order_name:  str,
        features:    np.ndarray,
    ) -> Dict[str, Any]:
        """Симуляция на основе предвычисленных данных."""
        order_key = order_name.lower().replace("grevlex", "grevlex")

        # Найти данные бенчмарка
        bench = RLEnvironment.BENCHMARK_TIMES
        if system_name and system_name in bench:
            times = bench[system_name]
        else:
            # Аппроксимация по сложности из вектора признаков
            complexity = features[7]
            base_time = 0.3 + complexity * 3.0
            times = {
                "lex":        base_time,
                "grevlex":    base_time * 0.55,
                "revgrevlex": base_time * 0.60,
            }

        order_key_lookup = order_name.lower()
        t_chosen = times.get(order_key_lookup, times.get("grevlex", 0.5))
        t_lex    = times.get("lex", t_chosen)
        speedup  = t_lex / max(t_chosen, 1e-9)

        # Имитация задержки (1 мс)
        time.sleep(0.001)

        return {
            "time_sec":      t_chosen,
            "speedup_vs_lex": speedup,
            "basis":         ["<simulation: basis not computed>"],
            "basis_length":  0,
            "reductions":    0,
        }

    # ─────────────────────────────────────────────────────────────────
    # Internal: real GInvDist computation
    # ─────────────────────────────────────────────────────────────────

    def _real_computation(
        self,
        polynomials: List,
        variables:   List,
        order_name:  str,
    ) -> Dict[str, Any]:
        """
        Реальное вычисление базиса через GInvDist API.
        Требует установленного пакета ginv.
        """
        if polynomials is None or variables is None:
            raise ValueError(
                "polynomials и variables обязательны для реального режима GInvDist."
            )

        n_vars = len(variables)
        permutation = ORDER_TO_PERMUTATION[order_name](n_vars)
        reversed_flag = (order_name == "REVGREVLEX")

        solver = ginv.GroebnerBasisSolver()                          # type: ignore
        solver.set_monomial_order_by_permutation(                    # type: ignore
            permutation=permutation,
            reversed=reversed_flag,
        )
        for poly in polynomials:
            solver.add_polynomial(str(poly))                         # type: ignore

        t_start = time.perf_counter()
        basis_raw = solver.compute()                                 # type: ignore
        t_end   = time.perf_counter()

        t_elapsed = t_end - t_start

        return {
            "time_sec":       t_elapsed,
            "speedup_vs_lex": 1.0,   # будет заполнено снаружи при сравнении
            "basis":          [str(b) for b in basis_raw],
            "basis_length":   len(basis_raw),
            "reductions":     getattr(solver, "n_reductions", 0),
        }

    # ─────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────

    def set_monomial_order_by_permutation(
        self,
        permutation: List[int],
        reversed_order: bool = False,
    ) -> None:
        """
        Низкоуровневый вызов GInvDist API.
        Устанавливает мономиальный порядок через перестановку переменных.

        Args:
            permutation:    список индексов переменных, задающий их порядок
            reversed_order: если True — обратный лексикографический порядок
        """
        if not GINVDIST_AVAILABLE:
            logger.warning(
                "set_monomial_order_by_permutation: GInvDist недоступен, вызов проигнорирован."
            )
            return
        solver = ginv.GroebnerBasisSolver()                          # type: ignore
        solver.set_monomial_order_by_permutation(                    # type: ignore
            permutation=permutation, reversed=reversed_order
        )
        logger.debug(
            "Установлен порядок: перестановка=%s, reversed=%s",
            permutation, reversed_order,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 65)
    print("GInvRLWrapper — демонстрация (режим симуляции)")
    print("=" * 65)

    wrapper = GInvRLWrapper.with_default_policy()

    for sys_name in ["cyclic_5", "eco_5", "chandra", "assur44"]:
        res = wrapper.solve(system_name=sys_name)
        probs = res["probabilities"]
        print(
            f"\n{sys_name:15s} → {res['order_chosen']:10s}  "
            f"t={res['time_sec']:.3f}s  "
            f"speedup={res['speedup_vs_lex']:.2f}x\n"
            f"  P(LEX)={probs['LEX']:.3f}  "
            f"P(GREVLEX)={probs['GREVLEX']:.3f}  "
            f"P(REVGREVLEX)={probs['REVGREVLEX']:.3f}"
        )

    print("\n\nСравнение всех порядков на cyclic_6:")
    all_results = wrapper.benchmark_all_orders(system_name="cyclic_6")
    for order, r in all_results.items():
        print(f"  {order:12s}: t={r['time_sec']:.3f}s  speedup={r['speedup_vs_lex']:.2f}x")
