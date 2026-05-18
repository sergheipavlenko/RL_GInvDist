"""
Model Architecture Visualization
==================================
Визуализация архитектуры нейронной сети с использованием torchinfo.
"""

import torch
import torch.nn as nn
from typing import Tuple
import json


class MonomorphicOrderPolicy(nn.Module):
    """
    Полносвязная нейронная сеть (MLP) для политики выбора порядка мономов.
    
    Архитектура:
    - Input layer: 8 признаков
    - Hidden layers: 128, 64
    - Output layer: 3 вероятности (для 3 типов порядков)
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


def get_torchinfo_summary():
    """
    Получить информацию об архитектуре используя torchinfo.
    """
    try:
        from torchinfo import summary
        
        model = MonomorphicOrderPolicy(state_dim=8, num_orders=3, hidden_dim=128)
        
        # Создание summary
        model_summary = summary(
            model,
            input_size=(1, 8),  # batch_size=1, feature_dim=8
            verbose=0,
            col_names=["input_size", "output_size", "num_params"],
        )
        
        return model_summary
    except ImportError:
        print("⚠️ torchinfo не установлен. Используем альтернативный метод.")
        return None


def generate_architecture_info():
    """
    Генерация информации об архитектуре в текстовом формате.
    """
    
    model = MonomorphicOrderPolicy(state_dim=8, num_orders=3, hidden_dim=128)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Информация о слоях
    layers_info = []
    
    # Policy network
    print("=" * 80)
    print("АРХИТЕКТУРА НЕЙРОННОЙ СЕТИ")
    print("=" * 80)
    
    print("\n📋 ПОЛИТИЧЕСКАЯ СЕТЬ (Policy Network):")
    print("-" * 80)
    
    policy_layers = list(model.policy_net.children())
    layer_num = 1
    input_features = 8
    
    for i, layer in enumerate(policy_layers):
        if isinstance(layer, nn.Linear):
            output_features = layer.out_features
            params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            
            print(f"  Слой {layer_num}: Linear(in={input_features}, out={output_features})")
            print(f"    - Параметров: {params:,}")
            print(f"    - Входной размер тензора: ({input_features},)")
            print(f"    - Выходной размер тензора: ({output_features},)")
            
            input_features = output_features
            layer_num += 1
        
        elif isinstance(layer, nn.ReLU):
            print(f"  Активация: ReLU")
            print(f"    - Входной размер: ({input_features},)")
            print(f"    - Выходной размер: ({input_features},)")
    
    print("\n📋 СЕТЬ ОЦЕНКИ СТОИМОСТИ (Value Network):")
    print("-" * 80)
    
    value_layers = list(model.value_net.children())
    layer_num = 1
    input_features = 8
    
    for i, layer in enumerate(value_layers):
        if isinstance(layer, nn.Linear):
            output_features = layer.out_features
            params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            
            print(f"  Слой {layer_num}: Linear(in={input_features}, out={output_features})")
            print(f"    - Параметров: {params:,}")
            print(f"    - Входной размер тензора: ({input_features},)")
            print(f"    - Выходной размер тензора: ({output_features},)")
            
            input_features = output_features
            layer_num += 1
        
        elif isinstance(layer, nn.ReLU):
            print(f"  Активация: ReLU")
            print(f"    - Входной размер: ({input_features},)")
            print(f"    - Выходной размер: ({input_features},)")
    
    print("\n" + "=" * 80)
    print("СВОДКА ПАРАМЕТРОВ МОДЕЛИ")
    print("=" * 80)
    
    print(f"\n📊 Общая статистика:")
    print(f"  • Всего параметров: {total_params:,}")
    print(f"  • Тренируемых параметров: {trainable_params:,}")
    print(f"  • Входная размерность: 8 (признаки полиномиальной системы)")
    print(f"  • Выходная размерность (действия): 3 (количество порядков)")
    print(f"  • Выходная размерность (оценка): 1 (значение состояния)")
    
    # Детальная статистика
    policy_params = sum(p.numel() for p in model.policy_net.parameters())
    value_params = sum(p.numel() for p in model.value_net.parameters())
    
    print(f"\n🔧 Параметры по подсетям:")
    print(f"  • Policy Network: {policy_params:,} параметров")
    print(f"  • Value Network: {value_params:,} параметров")
    print(f"  • Перекрытие слоев: {total_params - policy_params - value_params:,} параметров")
    
    # Информация о размерах тензоров
    print(f"\n📐 Размеры тензоров при одном примере:")
    print(f"  • Input: (1, 8) - [batch_size=1, features=8]")
    print(f"  • After Linear(8, 128): (1, 128)")
    print(f"  • After ReLU: (1, 128)")
    print(f"  • After Linear(128, 64): (1, 64)")
    print(f"  • After ReLU: (1, 64)")
    print(f"  • Policy Output Linear(64, 3): (1, 3)")
    print(f"  • Value Output Linear(64, 1): (1, 1)")
    
    # Информация о памяти
    print(f"\n💾 Приблизительное использование памяти:")
    bytes_per_param = 4  # float32
    memory_mb = (total_params * bytes_per_param) / (1024 * 1024)
    print(f"  • На весах: ~{memory_mb:.3f} МБ (float32)")
    
    return model, total_params


def generate_netron_compatible_format():
    """
    Генерация информации совместимой с Netron для визуализации.
    """
    
    model = MonomorphicOrderPolicy(state_dim=8, num_orders=3, hidden_dim=128)
    
    # Попытаемся экспортировать в ONNX для использования с Netron
    print("\n" + "=" * 80)
    print("ЭКСПОРТ МОДЕЛИ ДЛЯ NETRON")
    print("=" * 80)
    
    try:
        # Создание примера входа
        dummy_input = torch.randn(1, 8)
        
        # Экспорт в ONNX
        onnx_path = '/home/claude/rl_ginvdist_project/monomial_order_policy.onnx'
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['state'],
            output_names=['action_logits', 'value'],
            verbose=False,
            opset_version=12
        )
        
        print(f"\n✓ Модель экспортирована в ONNX: {onnx_path}")
        print("  Эту модель можно открыть в Netron (https://netron.app)")
        
    except Exception as e:
        print(f"\n⚠️ Не удалось экспортировать в ONNX: {e}")
        print("  Используйте web-версию Netron для визуализации")


def save_architecture_summary():
    """
    Сохранение информации об архитектуре в файл.
    """
    
    model = MonomorphicOrderPolicy(state_dim=8, num_orders=3, hidden_dim=128)
    
    # Информация об архитектуре в JSON
    architecture_info = {
        'model_name': 'MonomorphicOrderPolicy (MLP for Monomial Order Selection)',
        'framework': 'PyTorch',
        'input_shape': [1, 8],
        'output_shapes': {
            'action_logits': [1, 3],
            'value': [1, 1]
        },
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layers': [
            {
                'name': 'policy_net',
                'type': 'Sequential',
                'layers': [
                    {'type': 'Linear', 'in_features': 8, 'out_features': 128},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 128, 'out_features': 64},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 64, 'out_features': 3}
                ]
            },
            {
                'name': 'value_net',
                'type': 'Sequential',
                'layers': [
                    {'type': 'Linear', 'in_features': 8, 'out_features': 128},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 128, 'out_features': 64},
                    {'type': 'ReLU'},
                    {'type': 'Linear', 'in_features': 64, 'out_features': 1}
                ]
            }
        ],
        'description': 'MLP для обучения с подкреплением (PPO) выбора мономиального порядка в GInvDist'
    }
    
    with open('/home/claude/rl_ginvdist_project/architecture.json', 'w') as f:
        json.dump(architecture_info, f, indent=2)
    
    print("\n✓ Информация об архитектуре сохранена: architecture.json")


def main():
    """Основная функция."""
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ АРХИТЕКТУРЫ НЕЙРОННОЙ СЕТИ")
    print("=" * 80)
    
    # Генерация информации об архитектуре
    model, total_params = generate_architecture_info()
    
    # Попытка использования torchinfo
    try:
        import torchinfo
        print("\n[torchinfo] Получение детальной информации...")
        torchinfo_summary = get_torchinfo_summary()
        if torchinfo_summary:
            print(torchinfo_summary)
    except ImportError:
        print("\nℹ️  Для установки torchinfo выполните: pip install torchinfo")
    
    # Экспорт для Netron
    generate_netron_compatible_format()
    
    # Сохранение информации об архитектуре
    save_architecture_summary()
    
    print("\n" + "=" * 80)
    print("✓ АНАЛИЗ АРХИТЕКТУРЫ ЗАВЕРШЕН")
    print("=" * 80)


if __name__ == '__main__':
    main()
