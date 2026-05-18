"""
Reinforcement Learning Agent for Monomial Order Selection in GInvDist
=====================================================================
Агент обучения с подкреплением для автоматического выбора мономиального порядка
в библиотеке GInvDist с целью минимизации времени вычисления базиса Грёбнера.

Author: Pavlenko Sergey
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from collections import deque
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolynomialSystemFeatureExtractor:
    """
    Извлечение структурных признаков из полиномиальной системы.
    
    Признаки включают:
    - Количество переменных
    - Степень полинома
    - Плотность системы (отношение ненулевых коэффициентов)
    - Структурные индексы (количество мономов, среднюю степень)
    """
    
    def __init__(self, max_vars: int = 10, max_degree: int = 10):
        self.max_vars = max_vars
        self.max_degree = max_degree
        self.feature_dim = 8
    
    def extract_features(self, system_data: Dict) -> np.ndarray:
        """
        Извлечение признаков из системы.
        
        Args:
            system_data: Словарь с информацией о полиномиальной системе
                        (количество переменных, степени, размеры мономов и т.д.)
        
        Returns:
            Вектор признаков размера feature_dim
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # 1. Нормализованное количество переменных
        features[0] = min(system_data.get('num_vars', 1), self.max_vars) / self.max_vars
        
        # 2. Нормализованная максимальная степень
        max_deg = system_data.get('max_degree', 1)
        features[1] = min(max_deg, self.max_degree) / self.max_degree
        
        # 3. Средняя степень
        avg_deg = system_data.get('avg_degree', 1)
        features[2] = min(avg_deg, self.max_degree) / self.max_degree
        
        # 4. Количество полиномов
        num_polys = system_data.get('num_polynomials', 1)
        features[3] = min(num_polys, 50) / 50
        
        # 5. Среднее количество мономов в полиноме
        avg_monomials = system_data.get('avg_monomials', 1)
        features[4] = min(avg_monomials, 100) / 100
        
        # 6. Плотность системы (отношение ненулевых членов)
        density = system_data.get('density', 0.5)
        features[5] = np.clip(density, 0, 1)
        
        # 7. Спарсивность (противоположность плотности)
        features[6] = 1.0 - np.clip(density, 0, 1)
        
        # 8. Индекс сложности (составной признак)
        complexity = (max_deg * num_polys) / (avg_deg + 1)
        features[7] = min(complexity, 100) / 100
        
        return features


class MonomorphicOrderPolicy(nn.Module):
    """
    Полносвязная нейронная сеть (MLP) для политики выбора порядка мономов.
    
    Архитектура:
    - Input layer: 8 признаков
    - Hidden layers: 128, 64
    - Output layer: 3 вероятности (для 3 типов порядков)
    
    Типы порядков:
    1. Лексикографический (Lex)
    2. Степенно-лексикографический (GrevLex)  
    3. Степенно-обратно-лексикографический (Grevlex)
    """
    
    def __init__(self, state_dim: int = 8, num_orders: int = 3, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_orders = num_orders
        self.hidden_dim = hidden_dim
        
        # Policy network (для действий)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_orders)
        )
        
        # Value network (для оценки состояния) - для PPO
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов Xavier методом."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Тензор состояния размера (batch_size, state_dim) или (state_dim,)
        
        Returns:
            (action_logits, value): Логиты действий и оценка стоимости состояния
        """
        action_logits = self.policy_net(state)
        value = self.value_net(state)
        return action_logits, value
    
    def get_policy(self, state: torch.Tensor) -> dist.Categorical:
        """Получить распределение политики для выбора действия."""
        action_logits, _ = self.forward(state)
        return dist.Categorical(logits=action_logits)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Получить оценку стоимости состояния."""
        _, value = self.forward(state)
        return value


class RLEnvironment:
    """
    Окружение для тренировки RL-агента.
    Имитирует вычисление базиса Грёбнера с разными порядками мономов.
    """
    
    # Справочник известных вычислительных времен для разных систем и порядков
    BENCHMARK_TIMES = {
        'cyclic_5': {'lex': 0.85, 'grevlex': 0.42, 'revgrevlex': 0.45},
        'cyclic_6': {'lex': 2.10, 'grevlex': 1.15, 'revgrevlex': 1.25},
        'eco_5': {'lex': 0.22, 'grevlex': 0.15, 'revgrevlex': 0.18},
        'boon': {'lex': 0.55, 'grevlex': 0.28, 'revgrevlex': 0.32},
        'butcher': {'lex': 1.45, 'grevlex': 0.75, 'revgrevlex': 0.82},
        'chandra': {'lex': 3.20, 'grevlex': 1.85, 'revgrevlex': 2.05},
        'assur44': {'lex': 0.95, 'grevlex': 0.48, 'revgrevlex': 0.52},
        'oscillator_5': {'lex': 0.65, 'grevlex': 0.35, 'revgrevlex': 0.38},
    }
    
    ORDER_NAMES = ['lex', 'grevlex', 'revgrevlex']  # 0, 1, 2
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.current_system = None
    
    def reset(self, system_name: Optional[str] = None):
        """Сброс окружения на новую систему."""
        if system_name is None:
            system_name = self.rng.choice(list(self.BENCHMARK_TIMES.keys()))
        
        self.current_system = system_name
        
        # Имитация извлечения признаков
        features = self._generate_features(system_name)
        return features
    
    def _generate_features(self, system_name: str) -> np.ndarray:
        """Генерация признаков для системы."""
        # В реальности это будут извлеченные из полиномиальной системы признаки
        feature_extractor = PolynomialSystemFeatureExtractor()
        
        system_data = {
            'num_vars': self.rng.randint(3, 8),
            'max_degree': self.rng.randint(2, 6),
            'avg_degree': self.rng.uniform(2, 5),
            'num_polynomials': self.rng.randint(3, 8),
            'avg_monomials': self.rng.uniform(5, 30),
            'density': self.rng.uniform(0.3, 0.9),
        }
        
        return feature_extractor.extract_features(system_data)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Шаг окружения: выполнение действия (выбор порядка) и получение награды.
        
        Args:
            action: Индекс выбранного порядка (0=lex, 1=grevlex, 2=revgrevlex)
        
        Returns:
            (next_state, reward, done): Следующее состояние, награда, флаг окончания
        """
        order_name = self.ORDER_NAMES[action]
        exec_time = self.BENCHMARK_TIMES[self.current_system][order_name]
        
        # Оптимальный порядок (обычно grevlex)
        optimal_order_idx = 1  # grevlex
        optimal_time = self.BENCHMARK_TIMES[self.current_system][self.ORDER_NAMES[optimal_order_idx]]
        
        # Награда: ускорение относительно лучшего выбора
        # Нормализуем в диапазон [-1, 1]
        speedup = optimal_time / exec_time  # speedup > 1 если агент выбрал хорошо
        reward = np.clip((speedup - 1.0) / 2.0, -1.0, 1.0)
        
        # Штраф за плохой выбор
        if action != optimal_order_idx:
            reward -= 0.1
        
        next_state = self._generate_features(self.current_system)
        done = True  # Эпизод завершается после одного шага
        
        return next_state, reward, done


class PPOAgent:
    """
    Агент Proximal Policy Optimization для обучения политики выбора порядков.
    
    Гиперпараметры:
    - learning_rate: 3e-4
    - gamma: 0.99 (коэффициент дисконтирования)
    - gae_lambda: 0.95 (параметр GAE)
    - clip_ratio: 0.2 (клиппинг в PPO)
    - n_epochs: 3 (количество эпох для обновления)
    - batch_size: 64
    """
    
    def __init__(self, state_dim: int = 8, num_orders: int = 3, 
                 learning_rate: float = 3e-4, device: str = 'cpu'):
        self.state_dim = state_dim
        self.num_orders = num_orders
        self.device = torch.device(device)
        
        self.policy = MonomorphicOrderPolicy(state_dim, num_orders).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # PPO гиперпараметры
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.n_epochs = 3
        self.batch_size = 64
        
        # История для сбора траекторий
        self.trajectory_buffer = deque(maxlen=10000)
        
        logger.info(f"PPO Agent инициализирован на {device}")
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Выбор действия на основе текущей политики.
        
        Args:
            state: Вектор состояния
        
        Returns:
            Индекс выбранного порядка (0, 1 или 2)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy = self.policy.get_policy(state_tensor)
            action = policy.sample()
        return action.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float):
        """Сохранение переходов в буфер."""
        self.trajectory_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
    
    def train_batch(self, states: torch.Tensor, actions: torch.Tensor,
                   rewards: torch.Tensor, old_log_probs: torch.Tensor) -> Dict[str, float]:
        """
        Обновление политики на одном батче.
        
        Args:
            states: Батч состояний
            actions: Батч действий
            rewards: Батч (нормализованных) наград
            old_log_probs: Логарифмы вероятностей старой политики
        
        Returns:
            Словарь с метриками обучения
        """
        metrics = {'policy_loss': 0, 'value_loss': 0}
        
        for epoch in range(self.n_epochs):
            # Forward pass
            action_logits, values = self.policy(states)
            policy = dist.Categorical(logits=action_logits)
            
            # Логарифмы вероятностей новой политики
            new_log_probs = policy.log_prob(actions)
            entropy = policy.entropy().mean()
            
            # Ratio для PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate loss с клиппингом
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * rewards
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # Value loss
            value_loss = ((values.squeeze() - rewards) ** 2).mean()
            
            # Общая функция потерь
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
        
        # Усредняем по эпохам
        metrics['policy_loss'] /= self.n_epochs
        metrics['value_loss'] /= self.n_epochs
        
        return metrics
    
    def train(self, num_episodes: int = 100) -> List[Dict]:
        """
        Обучение агента на заданное количество эпизодов.
        
        Args:
            num_episodes: Количество эпизодов обучения
        
        Returns:
            Список истории награды по эпизодам
        """
        env = RLEnvironment()
        episode_rewards = []
        training_history = []
        
        logger.info(f"Начало обучения на {num_episodes} эпизодов...")
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            # Сбор эпизода
            while True:
                # Выбор действия
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy = self.policy.get_policy(state_tensor)
                    action = policy.sample()
                    log_prob = policy.log_prob(action).item()
                
                action = action.item()
                
                # Шаг в окружении
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                self.store_transition(state, action, reward, next_state, done, log_prob)
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Периодическое обновление политики
            if (episode + 1) % 10 == 0 and len(self.trajectory_buffer) >= self.batch_size:
                # Подготовка батча для обучения
                transitions = list(self.trajectory_buffer)
                states = torch.FloatTensor(np.array([t['state'] for t in transitions])).to(self.device)
                actions = torch.LongTensor([t['action'] for t in transitions]).to(self.device)
                rewards_list = torch.FloatTensor([t['reward'] for t in transitions]).to(self.device)
                old_log_probs = torch.FloatTensor([t['log_prob'] for t in transitions]).to(self.device)
                
                # Нормализация наград
                rewards_list = (rewards_list - rewards_list.mean()) / (rewards_list.std() + 1e-8)
                
                # Обучение
                metrics = self.train_batch(states, actions, rewards_list, old_log_probs)
                
                training_history.append({
                    'episode': episode,
                    'mean_reward': np.mean(episode_rewards[-10:]),
                    'policy_loss': metrics['policy_loss'],
                    'value_loss': metrics['value_loss']
                })
                
                logger.info(f"Episode {episode+1}/{num_episodes} | "
                           f"Mean Reward: {training_history[-1]['mean_reward']:.4f} | "
                           f"Policy Loss: {metrics['policy_loss']:.4f}")
        
        logger.info("Обучение завершено!")
        
        return training_history


def main():
    """Основная функция для демонстрации работы агента."""
    # Инициализация агента
    agent = PPOAgent(state_dim=8, num_orders=3, device='cpu')
    
    # Обучение
    history = agent.train(num_episodes=100)
    
    # Сохранение модели
    torch.save(agent.policy.state_dict(), '/home/claude/rl_ginvdist_project/model_weights.pth')
    
    # Сохранение истории обучения
    with open('/home/claude/rl_ginvdist_project/training_history.json', 'w') as f:
        # Конвертируем в JSON-сериализируемый формат
        json_history = [
            {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
             for k, v in item.items()}
            for item in history
        ]
        json.dump(json_history, f, indent=2)
    
    print("Model and history saved successfully!")


if __name__ == '__main__':
    main()
