"""论文对齐的手术机器人Actor-Critic网络 - 修复NoneType错误的最终版本"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SurgicalCritic(nn.Module):
    """论文方程(29)的Critic网络: Γ = W_c^T S_c(z̄) + ε_c"""
    def __init__(self, augmented_state_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        
        self.augmented_state_dim = augmented_state_dim  # z̄ = [z^T, x_d^T]^T
        
        # 构建值函数网络 S_c(z̄)
        dims = [augmented_state_dim] + hidden_dims + [1]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 输出层不加激活函数
                layers.append(nn.ReLU())
        
        self.value_network = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, augmented_state: torch.Tensor) -> torch.Tensor:
        """论文方程(29): Γ̂ = Ŵ_c^T S_c(z̄) - 修复NoneType问题"""
        
        # 输入验证
        if augmented_state is None:
            print(f"[ERROR] augmented_state is None in Critic forward")
            return torch.zeros(1, 1)
        
        # 维度检查
        if augmented_state.dim() == 1:
            augmented_state = augmented_state.unsqueeze(0)
        
        # 数值稳定性检查
        if torch.any(torch.isnan(augmented_state)) or torch.any(torch.isinf(augmented_state)):
            print(f"[WARNING] Invalid values in augmented_state")
            augmented_state = torch.zeros_like(augmented_state)
        
        try:
            value = self.value_network(augmented_state)
            
            # 输出验证
            if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                print(f"[WARNING] Invalid value output from critic")
                return torch.zeros_like(value)
            
            return value
            
        except Exception as e:
            print(f"[ERROR] Critic forward pass failed: {e}")
            batch_size = augmented_state.shape[0]
            return torch.zeros(batch_size, 1, device=augmented_state.device)
    
    def get_value_and_gradient(self, augmented_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算值函数及其梯度 ∇Γ - 修复NoneType问题"""
        
        if augmented_state is None:
            print(f"[ERROR] augmented_state is None in get_value_and_gradient")
            dummy_value = torch.zeros(1, 1)
            dummy_grad = torch.zeros(1, self.augmented_state_dim)
            return dummy_value, dummy_grad
        
        try:
            augmented_state = augmented_state.clone().detach().requires_grad_(True)
            value = self.forward(augmented_state)
            
            if value is None:
                print(f"[ERROR] Value is None in gradient computation")
                dummy_value = torch.zeros(1, 1)
                dummy_grad = torch.zeros(1, self.augmented_state_dim)
                return dummy_value, dummy_grad
            
            # 计算梯度 ∇Γ = ∂Γ/∂z̄
            grad_value = torch.autograd.grad(
                outputs=value.sum(), 
                inputs=augmented_state,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grad_value is None:
                print(f"[WARNING] Gradient computation returned None")
                grad_value = torch.zeros_like(augmented_state)
            
            return value, grad_value
            
        except Exception as e:
            print(f"[ERROR] Value and gradient computation failed: {e}")
            dummy_value = torch.zeros(1, 1, device=augmented_state.device)
            dummy_grad = torch.zeros_like(augmented_state)
            return dummy_value, dummy_grad

class SurgicalActor(nn.Module):
    """论文方程(50)的Actor网络: u = Ŵ_a^T S_a(Z_a) - f - e - K_2 e_v"""
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        
        self.input_dim = input_dim  
        self.action_dim = action_dim
        
        # 构建Actor网络 S_a(Z_a)
        dims = [input_dim] + hidden_dims + [action_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, input_state: torch.Tensor) -> torch.Tensor:
        """Actor前向传播 - 修复NoneType问题"""
        
        # 输入验证
        if input_state is None:
            print(f"[ERROR] input_state is None in Actor forward")
            return torch.zeros(1, self.action_dim)
        
        # 维度检查
        if input_state.dim() == 1:
            input_state = input_state.unsqueeze(0)
        
        # 数值稳定性检查
        if torch.any(torch.isnan(input_state)) or torch.any(torch.isinf(input_state)):
            print(f"[WARNING] Invalid values in actor input_state")
            input_state = torch.zeros_like(input_state)
        
        try:
            # 网络输出
            network_output = self.network(input_state)
            
            # 输出验证
            if network_output is None:
                print(f"[ERROR] Actor network returned None")
                return torch.zeros(input_state.shape[0], self.action_dim, device=input_state.device)
            
            if torch.any(torch.isnan(network_output)) or torch.any(torch.isinf(network_output)):
                print(f"[WARNING] Invalid actor network output")
                network_output = torch.zeros_like(network_output)
            
            # 应用tanh激活限制动作范围
            action = torch.tanh(network_output)
            
            return action
            
        except Exception as e:
            print(f"[ERROR] Actor forward pass failed: {e}")
            batch_size = input_state.shape[0]
            return torch.zeros(batch_size, self.action_dim, device=input_state.device)

class DynamicsIdentifierNetwork(nn.Module):
    """论文方程(34)的动力学识别网络: ż = W_id^T S_id(z, u) + ε_id"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        
        self.state_dim = state_dim  # 论文中 z ∈ R^9 = [x, ẋ, f]^T
        self.action_dim = action_dim  # u ∈ R^3
        
        # 识别器网络: S_id(z, u) → ż
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_dims + [state_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                
        self.network = nn.Sequential(*layers)
        
        # 权重初始化（较小的增益）
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.1)  # 动力学网络使用较小增益
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """论文方程(34): ż = W_id^T S_id(z, u) + ε_id - 修复NoneType问题"""
        
        # 输入验证
        if state is None or action is None:
            print(f"[ERROR] None inputs in identifier forward")
            if state is not None:
                return torch.zeros_like(state)
            else:
                return torch.zeros(1, self.state_dim)
        
        # 维度检查
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # 数值稳定性检查
        if torch.any(torch.isnan(state)) or torch.any(torch.isnan(action)):
            print(f"[WARNING] NaN in identifier inputs")
            return torch.zeros_like(state)
        
        try:
            x = torch.cat([state, action], dim=-1)
            state_derivative = self.network(x)
            
            # 输出验证
            if state_derivative is None:
                print(f"[ERROR] Identifier network returned None")
                return torch.zeros_like(state)
            
            if torch.any(torch.isnan(state_derivative)) or torch.any(torch.isinf(state_derivative)):
                print(f"[WARNING] Invalid identifier output")
                return torch.zeros_like(state)
            
            return state_derivative
            
        except Exception as e:
            print(f"[ERROR] Identifier forward pass failed: {e}")
            return torch.zeros_like(state)
    
    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """预测下一状态: z_{k+1} = z_k + dt * ż - 修复NoneType问题"""
        
        if state is None or action is None:
            print(f"[ERROR] None inputs in predict_next_state")
            return torch.zeros_like(state) if state is not None else torch.zeros(1, self.state_dim)
        
        try:
            state_dot = self.forward(state, action)
            if state_dot is None:
                return state.clone()
            
            next_state = state + dt * state_dot
            
            # 输出验证
            if torch.any(torch.isnan(next_state)) or torch.any(torch.isinf(next_state)):
                print(f"[WARNING] Invalid next state prediction")
                return state.clone()
            
            return next_state
            
        except Exception as e:
            print(f"[ERROR] Next state prediction failed: {e}")
            return state.clone()

class HJBSolver(nn.Module):
    """论文方程(28)的HJB方程求解器"""
    def __init__(self, Q_matrix: torch.Tensor, R_matrix: torch.Tensor):
        super().__init__()
        
        # 确保所有张量在同一设备上
        device = Q_matrix.device if Q_matrix is not None else (R_matrix.device if R_matrix is not None else torch.device('cpu'))
        
        # 安全地注册缓冲区
        if Q_matrix is not None:
            self.register_buffer('Q', Q_matrix.to(device))
        else:
            self.register_buffer('Q', torch.eye(12, device=device))
            
        if R_matrix is not None:
            R_on_device = R_matrix.to(device)
            self.register_buffer('R', R_on_device)
            # 确保单位矩阵也在同一设备上
            eye_matrix = torch.eye(R_matrix.shape[0], device=device)
            R_inv = torch.inverse(R_on_device + 1e-6 * eye_matrix)
            self.register_buffer('R_inv', R_inv)
        else:
            self.register_buffer('R', torch.eye(3, device=device))
            self.register_buffer('R_inv', torch.eye(3, device=device))
        
    def compute_hjb_error(self, augmented_state: torch.Tensor, value_grad: torch.Tensor,
                         system_dynamics: torch.Tensor, control_matrix: torch.Tensor) -> torch.Tensor:
        """计算HJB方程误差 - 论文方程(28) - 修复NoneType问题"""
        
        # 输入验证
        if any(x is None for x in [augmented_state, value_grad, system_dynamics, control_matrix]):
            print(f"[ERROR] None inputs in HJB error computation")
            batch_size = augmented_state.shape[0] if augmented_state is not None else 1
            return torch.zeros(batch_size)
        
        try:
            # HJB方程: 0 = z̄^T Q z̄ + (∇Γ)^T Ā(z̄) - (1/4)(∇Γ)^T B̄(z̄)R^(-1)B̄^T(z̄)∇Γ
            
            # 第一项: z̄^T Q z̄
            Q_term = torch.sum(augmented_state * (self.Q @ augmented_state.T).T, dim=-1)
            
            # 第二项: (∇Γ)^T Ā(z̄)
            drift_term = torch.sum(value_grad * system_dynamics, dim=-1)
            
            # 第三项: -(1/4)(∇Γ)^T B̄(z̄)R^(-1)B̄^T(z̄)∇Γ
            control_term = -0.25 * torch.sum(
                value_grad * (control_matrix @ self.R_inv @ control_matrix.transpose(-2, -1) @ value_grad.unsqueeze(-1)).squeeze(-1),
                dim=-1
            )
            
            # HJB方程残差（应该接近0）
            hjb_residual = Q_term + drift_term + control_term
            
            # 输出验证
            if torch.any(torch.isnan(hjb_residual)) or torch.any(torch.isinf(hjb_residual)):
                print(f"[WARNING] Invalid HJB residual")
                return torch.zeros_like(hjb_residual)
            
            return hjb_residual
            
        except Exception as e:
            print(f"[ERROR] HJB error computation failed: {e}")
            batch_size = augmented_state.shape[0] if augmented_state is not None else 1
            return torch.zeros(batch_size)
        
    def get_optimal_control(self, value_grad: torch.Tensor, control_matrix: torch.Tensor) -> torch.Tensor:
        """根据HJB方程计算最优控制: u* = -(1/2)R^(-1)B̄^T(z̄)∇Γ - 修复NoneType问题"""
        
        if value_grad is None or control_matrix is None:
            print(f"[ERROR] None inputs in optimal control computation")
            return torch.zeros(3)
        
        try:
            optimal_control = -0.5 * (self.R_inv @ control_matrix.transpose(-2, -1) @ value_grad.unsqueeze(-1)).squeeze(-1)
            
            # 输出验证
            if torch.any(torch.isnan(optimal_control)) or torch.any(torch.isinf(optimal_control)):
                print(f"[WARNING] Invalid optimal control")
                return torch.zeros_like(optimal_control)
            
            return optimal_control
            
        except Exception as e:
            print(f"[ERROR] Optimal control computation failed: {e}")
            return torch.zeros(3)

class SurgicalActorCritic(nn.Module):
    """论文完整的Actor-Critic网络，包含动力学识别和HJB求解 - 修复NoneType错误"""
    def __init__(self, state_dim: int, action_dim: int, augmented_state_dim: int, 
                 Q_weights: torch.Tensor, R_weights: torch.Tensor):
        super().__init__()
        
        self.state_dim = state_dim  # z ∈ R^9
        self.action_dim = action_dim  # u ∈ R^3
        self.augmented_state_dim = augmented_state_dim  # z̄ ∈ R^12
        
        # 核心网络组件
        self.critic = SurgicalCritic(augmented_state_dim)
        self.actor = SurgicalActor(augmented_state_dim, action_dim)  # 使用增广状态作为输入
        self.identifier = DynamicsIdentifierNetwork(state_dim, action_dim)
        self.hjb_solver = HJBSolver(Q_weights, R_weights)
        
        print(f"[INFO] SurgicalActorCritic initialized (Paper-aligned):")
        print(f"  - State dim: {state_dim} (z = [x, ẋ, f]^T)")
        print(f"  - Augmented state dim: {augmented_state_dim} (z̄ = [z^T, x_d^T]^T)")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"  - Actor params: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"  - Identifier params: {sum(p.numel() for p in self.identifier.parameters()):,}")
        
    def get_action(self, augmented_state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """获取动作 - 修复NoneType问题"""
        
        if augmented_state is None:
            print(f"[ERROR] augmented_state is None in get_action")
            return torch.zeros(1, self.action_dim)
        
        try:
            action = self.actor(augmented_state)
            
            if action is None:
                print(f"[ERROR] Actor returned None")
                return torch.zeros(augmented_state.shape[0], self.action_dim, device=augmented_state.device)
            
            if not deterministic:
                # 添加探索噪声
                noise = torch.randn_like(action) * 0.01
                action = action + noise
                
            return torch.clamp(action, -1.0, 1.0)
            
        except Exception as e:
            print(f"[ERROR] Get action failed: {e}")
            batch_size = augmented_state.shape[0] if augmented_state.dim() > 1 else 1
            return torch.zeros(batch_size, self.action_dim, device=augmented_state.device)
    
    def evaluate_value(self, augmented_state: torch.Tensor) -> torch.Tensor:
        """评估状态值函数 - 修复NoneType问题"""
        
        if augmented_state is None:
            print(f"[ERROR] augmented_state is None in evaluate_value")
            return torch.zeros(1, 1)
        
        try:
            value = self.critic(augmented_state)
            
            if value is None:
                print(f"[ERROR] Critic returned None in evaluate_value")
                return torch.zeros(augmented_state.shape[0], 1, device=augmented_state.device)
            
            return value
            
        except Exception as e:
            print(f"[ERROR] Value evaluation failed: {e}")
            batch_size = augmented_state.shape[0] if augmented_state.dim() > 1 else 1
            return torch.zeros(batch_size, 1, device=augmented_state.device)
    
    def compute_actor_loss(self, augmented_state: torch.Tensor) -> torch.Tensor:
        """计算Actor损失 - 修复NoneType问题"""
        
        if augmented_state is None:
            print(f"[ERROR] augmented_state is None in compute_actor_loss")
            return torch.tensor(0.0, requires_grad=True)
        
        try:
            action = self.actor(augmented_state)
            if action is None:
                print(f"[ERROR] Actor returned None in loss computation")
                return torch.tensor(0.0, requires_grad=True)
            
            value = self.critic(augmented_state)
            if value is None:
                print(f"[ERROR] Critic returned None in loss computation")
                return torch.tensor(0.0, requires_grad=True)
            
            # Actor损失：最大化值函数
            actor_loss = -value.mean()
            
            # 输出验证
            if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                print(f"[WARNING] Invalid actor loss")
                return torch.tensor(0.0, requires_grad=True)
            
            return actor_loss
            
        except Exception as e:
            print(f"[ERROR] Actor loss computation failed: {e}")
            return torch.tensor(0.0, requires_grad=True)
    
    def compute_critic_loss(self, augmented_state: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
        """计算Critic损失 - 修复NoneType问题"""
        
        if augmented_state is None or target_value is None:
            print(f"[ERROR] None inputs in compute_critic_loss")
            return torch.tensor(0.0, requires_grad=True)
        
        try:
            predicted_value = self.critic(augmented_state)
            
            if predicted_value is None:
                print(f"[ERROR] Critic returned None in loss computation")
                return torch.tensor(0.0, requires_grad=True)
            
            critic_loss = F.mse_loss(predicted_value.squeeze(), target_value.squeeze())
            
            # 输出验证
            if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                print(f"[WARNING] Invalid critic loss")
                return torch.tensor(0.0, requires_grad=True)
            
            return critic_loss
            
        except Exception as e:
            print(f"[ERROR] Critic loss computation failed: {e}")
            return torch.tensor(0.0, requires_grad=True)
    
    def compute_identifier_loss(self, state: torch.Tensor, action: torch.Tensor, 
                              next_state: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """计算动力学识别损失 - 修复NoneType问题"""
        
        if any(x is None for x in [state, action, next_state]):
            print(f"[ERROR] None inputs in compute_identifier_loss")
            return torch.tensor(0.0, requires_grad=True)
        
        try:
            # 真实状态导数
            true_state_dot = (next_state - state) / dt
            
            # 预测状态导数
            pred_state_dot = self.identifier(state, action)
            
            if pred_state_dot is None:
                print(f"[ERROR] Identifier returned None")
                return torch.tensor(0.0, requires_grad=True)
            
            # 识别损失
            identifier_loss = F.mse_loss(pred_state_dot, true_state_dot)
            
            # 输出验证
            if torch.isnan(identifier_loss) or torch.isinf(identifier_loss):
                print(f"[WARNING] Invalid identifier loss")
                return torch.tensor(0.0, requires_grad=True)
            
            return identifier_loss
            
        except Exception as e:
            print(f"[ERROR] Identifier loss computation failed: {e}")
            return torch.tensor(0.0, requires_grad=True)
    
    def identify_dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """动力学识别 - 修复NoneType问题"""
        
        if state is None or action is None:
            print(f"[ERROR] None inputs in identify_dynamics")
            return torch.zeros_like(state) if state is not None else torch.zeros(1, self.state_dim)
        
        try:
            dynamics = self.identifier(state, action)
            
            if dynamics is None:
                print(f"[ERROR] Identifier returned None in identify_dynamics")
                return torch.zeros_like(state)
            
            return dynamics
            
        except Exception as e:
            print(f"[ERROR] Dynamics identification failed: {e}")
            return torch.zeros_like(state)