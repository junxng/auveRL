import unittest
import numpy as np
import os
import sys
import tempfile
import torch

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn_model import DQNAgent

class TestDQNAgent(unittest.TestCase):
    """
    Kiểm thử DQN Agent với PyTorch
    """
    
    def setUp(self):
        """
        Cài đặt môi trường cho mỗi test case
        """
        self.state_size = 14  # Giống như trong môi trường
        self.action_size = 5  # Số hành động có thể
        self.device = torch.device("cpu")  # Sử dụng CPU cho kiểm thử
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            memory_size=100,
            batch_size=16,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            device=self.device
        )
    
    def test_network_architecture(self):
        """
        Kiểm tra kiến trúc mạng neural
        """
        # Kiểm tra model có phải là PyTorch module
        self.assertIsInstance(self.agent.model, torch.nn.Module)
        self.assertIsInstance(self.agent.target_model, torch.nn.Module)
        
        # Tạo input tensor giả
        test_input = torch.randn(1, self.state_size).to(self.device)
        
        # Kiểm tra forward pass
        with torch.no_grad():
            output = self.agent.model(test_input)
        
        # Kiểm tra kích thước đầu ra
        self.assertEqual(output.shape, (1, self.action_size))
    
    def test_act(self):
        """
        Kiểm tra hàm act
        """
        state = np.random.random((self.state_size,))
        
        # Kiểm tra trong chế độ huấn luyện (có thăm dò)
        action = self.agent.act(state, training=True)
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < self.action_size)
        
        # Kiểm tra trong chế độ inference (không thăm dò)
        self.agent.epsilon = 0  # Đảm bảo không thăm dò
        action = self.agent.act(state, training=False)
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < self.action_size)
    
    def test_remember(self):
        """
        Kiểm tra hàm lưu trữ kinh nghiệm
        """
        state = np.random.random((self.state_size,))
        action = 0
        reward = 1.0
        next_state = np.random.random((self.state_size,))
        done = False
        
        # Lưu trữ một kinh nghiệm
        self.agent.remember(state, action, reward, next_state, done)
        
        # Kiểm tra kinh nghiệm đã được lưu
        self.assertEqual(len(self.agent.memory), 1)
        
        # Kiểm tra thành phần của kinh nghiệm
        experience = self.agent.memory[0]
        self.assertIsInstance(experience, tuple)
        self.assertEqual(len(experience), 5)
        
        # Lưu trữ nhiều kinh nghiệm để kiểm tra memory size
        for _ in range(110):  # Lớn hơn memory_size
            self.agent.remember(state, action, reward, next_state, done)
        
        # Kiểm tra kích thước memory không vượt quá giới hạn
        self.assertEqual(len(self.agent.memory), 100)
    
    def test_replay(self):
        """
        Kiểm tra hàm replay (huấn luyện từ memory)
        """
        # Thêm đủ kinh nghiệm vào memory
        state = np.random.random((self.state_size,))
        next_state = np.random.random((self.state_size,))
        
        for _ in range(20):  # Nhiều hơn batch_size
            action = np.random.randint(0, self.action_size)
            reward = np.random.random()
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)
        
        # Lưu epsilon hiện tại
        old_epsilon = self.agent.epsilon
        
        # Thực hiện replay
        loss = self.agent.replay()
        
        # Kiểm tra kết quả replay
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        
        # Kiểm tra epsilon đã giảm
        self.assertLess(self.agent.epsilon, old_epsilon)
    
    def test_save_load(self):
        """
        Kiểm tra hàm lưu và tải mô hình
        """
        # Tạo file tạm thời để lưu mô hình
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Lưu mô hình
            self.agent.save(tmp.name)
            
            # Tạo agent mới
            new_agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                device=self.device
            )
            
            # Tải mô hình từ file
            success = new_agent.load(tmp.name)
            
            # Kiểm tra tải thành công
            self.assertTrue(success)
            
            # Kiểm tra state_dict của model có cùng tên các khóa
            old_keys = set(self.agent.model.state_dict().keys())
            new_keys = set(new_agent.model.state_dict().keys())
            self.assertEqual(old_keys, new_keys)
            
            # Kiểm tra epsilon đã được lưu và tải đúng
            self.assertEqual(self.agent.epsilon, new_agent.epsilon)
    
    def test_update_target_model(self):
        """
        Kiểm tra hàm cập nhật target model
        """
        # Thay đổi một số trọng số trong model
        for param in self.agent.model.parameters():
            param.data += 0.1
        
        # Trọng số của model và target_model bây giờ khác nhau
        for p1, p2 in zip(self.agent.model.parameters(), self.agent.target_model.parameters()):
            # Kiểm tra ít nhất một tham số khác nhau
            if not torch.allclose(p1.data, p2.data):
                break
        else:
            self.fail("Không tìm thấy sự khác biệt giữa model và target_model")
        
        # Cập nhật target model
        self.agent.update_target_model()
        
        # Kiểm tra trọng số đã được cập nhật
        for p1, p2 in zip(self.agent.model.parameters(), self.agent.target_model.parameters()):
            self.assertTrue(torch.allclose(p1.data, p2.data))
    
    def tearDown(self):
        """
        Dọn dẹp sau mỗi test case
        """
        pass

if __name__ == "__main__":
    unittest.main()