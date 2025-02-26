import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
from collections import deque

class DQNNetwork(nn.Module):
    """
    Mạng neural cho Deep Q-Network
    """
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        # Cải thiện kiến trúc mạng
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)  # Thêm dropout để giảm overfitting
            ])
            prev_size = hidden_size
        
        # Layer cuối cùng không có BatchNorm hoặc ReLU
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)

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

class DQNAgent:
    """
    Đại diện DQN (Deep Q-Network) cho xe tự hành.
    """
    def __init__(
        self,
        state_size,
        action_size,
        memory_size=10000,
        batch_size=64,
        gamma=0.95,  # Giảm gamma xuống để giảm tầm nhìn xa
        epsilon=1.0,
        epsilon_min=0.05,  # Tăng epsilon min để duy trì đủ thăm dò
        epsilon_decay=0.99,  # Giảm tốc độ giảm epsilon
        learning_rate=0.0005,  # Giảm learning rate
        target_update_freq=10,  # Tăng tần số cập nhật mạng mục tiêu
        double_dqn=True,  # Sử dụng Double DQN
        device=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma  # Hệ số chiết khấu
        self.epsilon = epsilon  # Tham số thăm dò
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.double_dqn = double_dqn
        self.training_steps = 0
        
        # Xác định device (CPU hoặc GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Tạo các mạng neural
        self.model = DQNNetwork(state_size, action_size).to(self.device)  # Mạng chính
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)  # Mạng mục tiêu
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function - Huber Loss (smooth L1)
        self.criterion = nn.SmoothL1Loss()
        
        # Đồng bộ trọng số ban đầu
        self.update_target_model()
    
    def update_target_model(self):
        """
        Cập nhật trọng số từ mạng chính sang mạng mục tiêu.
        """
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Lưu trữ kinh nghiệm vào bộ nhớ.
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            training: Boolean cho biết có đang huấn luyện không
            
        Returns:
            action: Hành động được chọn
        """
        # Trong quá trình huấn luyện, thực hiện thăm dò epsilon-greedy
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Chuyển đổi state thành tensor và đưa vào device
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Đặt mô hình ở chế độ đánh giá khi dự đoán
        self.model.eval()
        
        with torch.no_grad():
            q_values = self.model(state)
        
        # Đặt mô hình về chế độ huấn luyện nếu cần
        if training:
            self.model.train()
        
        return q_values.cpu().data.numpy()[0].argmax()
    
    def replay(self):
        """
        Huấn luyện mô hình dựa trên kinh nghiệm đã lưu.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Lấy ngẫu nhiên một batch từ bộ nhớ
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Chuẩn bị dữ liệu
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Chuyển đổi thành tensor và đưa vào device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Tính toán Q-values hiện tại cho các trạng thái
        self.model.train()
        q_values = self.model(states)
        
        # Lấy Q-values cho các hành động đã chọn
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Tính toán Q-values mục tiêu
        with torch.no_grad():
            self.target_model.eval()
            
            if self.double_dqn:
                # Double DQN: Sử dụng mạng chính để chọn hành động và mạng mục tiêu để đánh giá
                self.model.eval()
                next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
                self.model.train()
                next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN: Sử dụng mạng mục tiêu để cả chọn và đánh giá hành động
                next_q_values = self.target_model(next_states).max(1)[0]
            
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Tính loss và thực hiện backpropagation
        loss = self.criterion(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradient để ổn định học tập
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Tăng bộ đếm bước huấn luyện
        self.training_steps += 1
        
        # Giảm epsilon sau mỗi lần huấn luyện
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Cập nhật mạng mục tiêu sau một số bước nhất định
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()
            self.update_counter = 0
        
        return loss.item()
    
    def load(self, path):
        """
        Tải trọng số mô hình.
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            if 'training_steps' in checkpoint:
                self.training_steps = checkpoint['training_steps']
                
            # Đặt mạng ở chế độ eval sau khi tải để tránh vấn đề với BatchNorm
            self.model.eval()
            self.target_model.eval()
                
            print(f"Mô hình DQN được tải từ {path}")
            return True
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return False
    
    def save(self, path):
        """
        Lưu trọng số mô hình.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_steps': self.training_steps
            }, path)
            print(f"Mô hình DQN được lưu tại {path}")
            return True
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {e}")
            return False