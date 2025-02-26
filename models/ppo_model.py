import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os

class ActorNetwork(nn.Module):
    """
    Mạng Actor cho PPO
    """
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Đặt tất cả BatchNorm layers sang eval mode khi batch_size=1
        if x.size(0) == 1:
            training_state = self.training
            self.eval()
            output = self.network(x)
            if training_state:
                self.train()
            return output
        else:
            return self.network(x)

class CriticNetwork(nn.Module):
    """
    Mạng Critic cho PPO
    """
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Đặt tất cả BatchNorm layers sang eval mode khi batch_size=1
        if x.size(0) == 1:
            training_state = self.training
            self.eval()
            output = self.network(x)
            if training_state:
                self.train()
            return output
        else:
            return self.network(x)

class PPOAgent:
    """
    Đại diện PPO (Proximal Policy Optimization) cho xe tự hành.
    """
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        clip_ratio=0.2,
        policy_lr=0.0003,
        value_lr=0.001,
        batch_size=64,
        epochs=10,
        lmbda=0.95,
        device=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Hệ số chiết khấu
        self.clip_ratio = clip_ratio  # Epsilon cho clipping
        self.policy_lr = policy_lr  # Learning rate cho mạng policy
        self.value_lr = value_lr  # Learning rate cho mạng value
        self.batch_size = batch_size
        self.epochs = epochs
        self.lmbda = lmbda  # Tham số lambda cho tính GAE
        
        # Xác định device (CPU hoặc GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Tạo mạng Actor và Critic
        self.actor = ActorNetwork(state_size, action_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=value_lr)
        
        # Bộ nhớ cho dữ liệu huấn luyện
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def act(self, state, evaluation=False):
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            evaluation: Boolean cho biết có đang đánh giá mô hình hay không
        
        Returns:
            action: Hành động được chọn
            value: Giá trị trạng thái từ critic
            log_prob: Log probability của hành động
        """
        # Chuyển đổi state thành tensor và đưa vào device
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Đặt mạng ở chế độ đánh giá nếu cần
        if evaluation:
            self.actor.eval()
            self.critic.eval()
        
        with torch.no_grad():
            # Lấy phân phối xác suất từ actor
            action_probs = self.actor(state)
            
            # Tạo phân phối categorical
            dist = Categorical(action_probs)
            
            # Lấy mẫu hành động từ phân phối nếu đang train, hoặc hành động tốt nhất nếu đang evaluate
            if evaluation:
                action = torch.argmax(action_probs, dim=1)
            else:
                action = dist.sample()
            
            # Tính log probability của hành động
            log_prob = dist.log_prob(action)
            
            # Lấy giá trị trạng thái từ critic
            value = self.critic(state)
        
        # Trở về chế độ huấn luyện nếu đang train
        if not evaluation:
            self.actor.train()
            self.critic.train()
        
        return action.item(), value.item(), log_prob.item()
    
    def remember(self, state, action, reward, value, log_prob, done):
        """
        Lưu trữ kinh nghiệm vào bộ nhớ.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def _compute_advantages(self, values, dones, rewards):
        """
        Tính toán advantages sử dụng GAE (Generalized Advantage Estimation).
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        
        # Đi ngược từ cuối để tính giá trị
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Trạng thái cuối cùng
                if dones[t]:
                    # Nếu là trạng thái kết thúc, next_value = 0
                    next_value = 0
                else:
                    # Dùng bootstrap từ value function
                    next_state = np.reshape(self.states[t], [1, self.state_size])
                    next_state = torch.FloatTensor(next_state).to(self.device)
                    with torch.no_grad():
                        self.critic.eval()
                        next_value = self.critic(next_state).item()
                        self.critic.train()
            else:
                # Sử dụng giá trị đã tính
                next_value = values[t + 1]
                
            # Delta = R + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE = delta + gamma * lambda * GAE từ trước (nếu không phải trạng thái kết thúc)
            gae = delta + self.gamma * self.lmbda * gae * (1 - dones[t])
            
            # Lưu advantage
            advantages[t] = gae
            
        # Tính returns (cho critic training)
        returns = advantages + np.array(values)
        
        return advantages, returns
    
    def train(self):
        """
        Huấn luyện mô hình theo thuật toán PPO.
        """
        if len(self.states) == 0:
            return
        
        # Chuyển các lists thành arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        log_probs = np.array(self.log_probs)
        dones = np.array(self.dones)
        
        # Tính advantages và returns
        advantages, returns = self._compute_advantages(values, dones, rewards)
        
        # Chuẩn hóa advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Chuyển đổi thành tensor và đưa vào device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Số lượng mẫu và batch
        n_samples = len(states)
        n_batches = n_samples // self.batch_size
        
        # Huấn luyện qua nhiều epochs
        for _ in range(self.epochs):
            # Tạo indices và xáo trộn
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            # Huấn luyện qua các batches
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Lấy batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Tính log probabilities mới và values
                self.actor.train()
                action_probs = self.actor(batch_states)
                dist = Categorical(action_probs)
                batch_new_log_probs = dist.log_prob(batch_actions)
                
                self.critic.train()
                batch_values = self.critic(batch_states).squeeze(1)
                
                # Tính ratio for PPO
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                
                # Tính surrogate objectives
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * batch_advantages
                
                # Tính actor (policy) loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Tính critic (value) loss
                critic_loss = nn.MSELoss()(batch_values, batch_returns)
                
                # Backpropagation cho actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Backpropagation cho critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        
        # Xóa bộ nhớ sau khi huấn luyện
        self.clear_memory()
    
    def clear_memory(self):
        """
        Xóa bộ nhớ sau mỗi episode.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def load(self, actor_path, critic_path):
        """
        Tải trọng số mô hình.
        """
        try:
            actor_checkpoint = torch.load(actor_path, map_location=self.device)
            critic_checkpoint = torch.load(critic_path, map_location=self.device)
            
            self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
            self.critic.load_state_dict(critic_checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
            
            # Đặt mạng ở chế độ eval để tránh vấn đề với BatchNorm
            self.actor.eval()
            self.critic.eval()
            
            print(f"Mô hình PPO được tải từ {actor_path} và {critic_path}")
            return True
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return False
    
    def save(self, actor_path, critic_path):
        """
        Lưu trọng số mô hình.
        """
        try:
            os.makedirs(os.path.dirname(actor_path), exist_ok=True)
            os.makedirs(os.path.dirname(critic_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': self.actor.state_dict(),
                'optimizer_state_dict': self.actor_optimizer.state_dict(),
            }, actor_path)
            
            torch.save({
                'model_state_dict': self.critic.state_dict(),
                'optimizer_state_dict': self.critic_optimizer.state_dict(),
            }, critic_path)
            
            print(f"Mô hình PPO được lưu tại {actor_path} và {critic_path}")
            return True
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {e}")
            return False