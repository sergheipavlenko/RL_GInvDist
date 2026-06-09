"""
feature_extractor.py
=====================
Модуль извлечения структурных признаков полиномиальной системы.

Извлекает 8-мерный вектор признаков из полиномиальной системы,
используемый как вектор состояния RL-агента.

Author: Pavlenko Sergey
Group:  НПИбд-02-23, RUDN University
Date:   2026
"""

import numpy as np
from typing import Dict, List, Optional, Union

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


class PolynomialSystemFeatureExtractor:
    """
    Извлечение 8 структурных признаков из полиномиальной системы.

    Признаки:
        [0] n_vars_norm       — нормализованное число переменных
        [1] max_degree_norm   — нормализованная максимальная степень
        [2] avg_degree_norm   — нормализованная средняя степень
        [3] n_polys_norm      — нормализованное число полиномов
        [4] avg_monomials_norm— нормализованное среднее число мономов
        [5] density           — плотность (доля ненулевых коэффициентов)
        [6] sparsity          — спарсивность (1 - density)
        [7] complexity_index  — индекс сложности (оценка числа S-пар)

    Все признаки нормализованы в [0, 1].
    """

    FEATURE_NAMES = [
        "n_vars_norm",
        "max_degree_norm",
        "avg_degree_norm",
        "n_polys_norm",
        "avg_monomials_norm",
        "density",
        "sparsity",
        "complexity_index",
    ]
    FEATURE_DIM = 8

    def __init__(self, max_vars: int = 10, max_degree: int = 20,
                 max_polys: int = 50, max_monomials: int = 200):
        self.max_vars = max_vars
        self.max_degree = max_degree
        self.max_polys = max_polys
        self.max_monomials = max_monomials

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, polynomials=None, variables=None,
                system_dict: Optional[Dict] = None) -> np.ndarray:
        """
        Извлечь вектор признаков.

        Может принимать либо SymPy-объекты (polynomials + variables),
        либо словарь с предвычисленными полями (system_dict).

        Args:
            polynomials: список SymPy-полиномов (если SymPy доступен)
            variables:   список SymPy-символов
            system_dict: словарь с полями num_vars, max_degree, avg_degree,
                         num_polynomials, avg_monomials, density

        Returns:
            np.ndarray shape (8,) dtype float32
        """
        if system_dict is not None:
            return self._from_dict(system_dict)
        if polynomials is not None and SYMPY_AVAILABLE:
            return self._from_sympy(polynomials, variables or [])
        raise ValueError(
            "Передайте либо system_dict, либо polynomials (с установленным SymPy)."
        )

    def extract_batch(self, systems: List[Dict]) -> np.ndarray:
        """
        Пакетное извлечение признаков из списка словарей.

        Returns:
            np.ndarray shape (N, 8) dtype float32
        """
        return np.stack([self._from_dict(s) for s in systems]).astype(np.float32)

    def feature_names(self) -> List[str]:
        return list(self.FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _from_dict(self, d: Dict) -> np.ndarray:
        """Извлечение признаков из словаря с предвычисленными данными."""
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        n_vars = float(d.get("num_vars", 1))
        max_deg = float(d.get("max_degree", 1))
        avg_deg = float(d.get("avg_degree", max_deg))
        n_polys = float(d.get("num_polynomials", 1))
        avg_mon = float(d.get("avg_monomials", 1))
        density = float(d.get("density", 0.5))

        features[0] = min(n_vars, self.max_vars) / self.max_vars
        features[1] = min(max_deg, self.max_degree) / self.max_degree
        features[2] = min(avg_deg, self.max_degree) / self.max_degree
        features[3] = min(n_polys, self.max_polys) / self.max_polys
        features[4] = min(avg_mon, self.max_monomials) / self.max_monomials
        features[5] = float(np.clip(density, 0.0, 1.0))
        features[6] = 1.0 - features[5]
        # Индекс сложности: оценка числа S-пар, нормализованная
        complexity = (n_polys ** 2) * (max_deg + 1) / 1000.0
        features[7] = float(min(complexity, 10.0) / 10.0)

        return features

    def _from_sympy(self, polynomials, variables) -> np.ndarray:
        """Извлечение признаков непосредственно из SymPy-полиномов."""
        if not polynomials:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        n_vars = len(variables) if variables else _count_free_symbols(polynomials)
        degrees = [int(sp.total_degree(p, *variables)) if variables
                   else int(sp.total_degree(p)) for p in polynomials]
        max_deg = max(degrees) if degrees else 1
        avg_deg = float(np.mean(degrees)) if degrees else 1.0
        n_polys = len(polynomials)

        n_monomials = [len(p.as_ordered_terms()) for p in polynomials]
        avg_mon = float(np.mean(n_monomials)) if n_monomials else 1.0

        # Оценка плотности: доля ненулевых коэффициентов относительно
        # числа мономов максимальной степени
        max_possible = sum(
            _binomial(d + n_vars, n_vars) for d in range(max_deg + 1)
        ) if n_vars > 0 else 1
        density = min(avg_mon / max(max_possible, 1), 1.0)

        return self._from_dict({
            "num_vars": n_vars,
            "max_degree": max_deg,
            "avg_degree": avg_deg,
            "num_polynomials": n_polys,
            "avg_monomials": avg_mon,
            "density": density,
        })


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _count_free_symbols(polynomials) -> int:
    """Подсчёт числа уникальных переменных в системе."""
    symbols = set()
    for p in polynomials:
        symbols |= p.free_symbols
    return len(symbols)


def _binomial(n: int, k: int) -> int:
    """Биномиальный коэффициент C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


# ------------------------------------------------------------------
# Predefined benchmark system descriptors
# ------------------------------------------------------------------

BENCHMARK_SYSTEMS: Dict[str, Dict] = {
    "cyclic_5": {
        "num_vars": 5, "max_degree": 5, "avg_degree": 4.2,
        "num_polynomials": 5, "avg_monomials": 28.4, "density": 0.71,
        "description": "Cyclic-5 — классический тест для алгоритмов базиса Грёбнера",
    },
    "cyclic_6": {
        "num_vars": 6, "max_degree": 6, "avg_degree": 5.1,
        "num_polynomials": 6, "avg_monomials": 42.7, "density": 0.68,
        "description": "Cyclic-6 — более сложный вариант циклической системы",
    },
    "eco_5": {
        "num_vars": 5, "max_degree": 3, "avg_degree": 2.6,
        "num_polynomials": 5, "avg_monomials": 8.2, "density": 0.45,
        "description": "eco_5 — система экономической кинетики",
    },
    "boon": {
        "num_vars": 7, "max_degree": 4, "avg_degree": 3.4,
        "num_polynomials": 5, "avg_monomials": 18.6, "density": 0.52,
        "description": "boon — классический тест среднего уровня",
    },
    "butcher": {
        "num_vars": 8, "max_degree": 5, "avg_degree": 4.1,
        "num_polynomials": 7, "avg_monomials": 31.4, "density": 0.58,
        "description": "butcher — задача Butcher, часто используется для бенчмарков",
    },
    "chandra": {
        "num_vars": 6, "max_degree": 8, "avg_degree": 6.3,
        "num_polynomials": 8, "avg_monomials": 47.8, "density": 0.79,
        "description": "chandra — система Chandrasekhar высокой сложности",
    },
    "assur44": {
        "num_vars": 8, "max_degree": 4, "avg_degree": 3.2,
        "num_polynomials": 6, "avg_monomials": 22.5, "density": 0.49,
        "description": "assur44 — система из робототехники (механизм Assur)",
    },
    "oscillator_5": {
        "num_vars": 5, "max_degree": 3, "avg_degree": 2.8,
        "num_polynomials": 1, "avg_monomials": 11.0, "density": 0.44,
        "description": "oscillator_5 — нелинейный кубический осциллятор",
    },
}


if __name__ == "__main__":
    extractor = PolynomialSystemFeatureExtractor()
    print("=" * 60)
    print("PolynomialSystemFeatureExtractor — демонстрация")
    print("=" * 60)
    print(f"\nРазмерность признакового вектора: {extractor.FEATURE_DIM}")
    print(f"Имена признаков: {extractor.feature_names()}\n")

    for name, sdata in BENCHMARK_SYSTEMS.items():
        feats = extractor.extract(system_dict=sdata)
        print(f"{name:15s}: {np.round(feats, 3)}")
