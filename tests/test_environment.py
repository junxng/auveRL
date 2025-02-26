import unittest
import numpy as np
import os
import sys

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.urban_environment import UrbanEnvironment

class TestUrbanEnvironment(unittest.TestCase):
    """
    Kiểm thử môi trường đô thị
    """
    
    def setUp(self):
        """
        Cài đặt môi trường cho mỗi test case
        """
        self.grid_size = 50
        self.num_obstacles = 10
        self.env = UrbanEnvironment(
            grid_size=self.grid_size,
            num_obstacles=self.num_obstacles,
            dynamic_obstacles=False
        )
    
    def test_reset(self):
        """
        Kiểm tra hàm reset
        """
        state, info = self.env.reset(seed=42)  # Cập nhật API reset
        
        # Kiểm tra kích thước state
        self.assertEqual(len(state), self.env.observation_space.shape[0])
        
        # Kiểm tra vị trí xe trong khoảng hợp lệ
        self.assertTrue(0 <= self.env.car_pos[0] < self.grid_size)
        self.assertTrue(0 <= self.env.car_pos[1] < self.grid_size)
        
        # Kiểm tra vị trí đích trong khoảng hợp lệ
        self.assertTrue(0 <= self.env.target_pos[0] < self.grid_size)
        self.assertTrue(0 <= self.env.target_pos[1] < self.grid_size)
        
        # Kiểm tra số lượng chướng ngại vật
        self.assertEqual(len(self.env.obstacles), self.num_obstacles)
        
        # Kiểm tra info là dict
        self.assertIsInstance(info, dict)
    
    def test_step(self):
        """
        Kiểm tra hàm step
        """
        self.env.reset(seed=42)
        
        # Thực hiện một hành động hợp lệ
        next_state, reward, terminated, truncated, info = self.env.step(0)  # Cập nhật API step
        
        # Kiểm tra kích thước next_state
        self.assertEqual(len(next_state), self.env.observation_space.shape[0])
        
        # Kiểm tra reward là số thực
        self.assertIsInstance(reward, (int, float))
        
        # Kiểm tra terminated và truncated là boolean
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        
        # Kiểm tra info là dictionary
        self.assertIsInstance(info, dict)
        
        # Kiểm tra vận tốc xe đã thay đổi sau khi tăng tốc
        self.assertTrue(np.any(self.env.car_velocity > 0))
    
    def test_out_of_bounds(self):
        """
        Kiểm tra xe ra khỏi biên
        """
        self.env.reset(seed=42)
        
        # Di chuyển xe đến gần biên
        self.env.car_pos = np.array([1.0, 1.0])
        self.env.car_velocity = np.array([-0.5, -0.5])
        
        # Thực hiện hành động tăng tốc về phía biên
        _, reward, terminated, truncated, _ = self.env.step(0)
        
        # Xe nên va chạm với biên và kết thúc episode
        self.assertTrue(terminated)  # Cập nhật từ done thành terminated
        self.assertLess(reward, 0)  # Phần thưởng âm khi va chạm
    
    def test_sensor_data(self):
        """
        Kiểm tra dữ liệu cảm biến
        """
        self.env.reset(seed=42)
        
        # Đặt vị trí xe và chướng ngại vật cố định
        self.env.car_pos = np.array([25.0, 25.0])
        self.env.obstacles = [np.array([35.0, 25.0])]  # Chướng ngại vật ở phía đông xe
        
        # Lấy dữ liệu cảm biến
        sensor_data = self.env._get_sensor_data()
        
        # Cảm biến hướng đông (index 0) nên phát hiện chướng ngại vật gần
        self.assertLess(sensor_data[0], self.grid_size)
        
        # Các cảm biến khác nên không phát hiện chướng ngại vật gần
        for i in range(1, 8):
            self.assertGreater(sensor_data[i], 10)
    
    def test_target_reached(self):
        """
        Kiểm tra khi xe đến đích
        """
        self.env.reset(seed=42)
        
        # Đặt xe gần đích
        self.env.car_pos = self.env.target_pos - np.array([1.0, 1.0])
        self.env.car_velocity = np.array([0.5, 0.5])
        
        # Thực hiện hành động tăng tốc về phía đích
        _, reward, terminated, truncated, _ = self.env.step(0)
        
        # Xe nên đến đích và kết thúc episode với phần thưởng lớn
        self.assertTrue(terminated)  # Cập nhật từ done thành terminated
        self.assertGreater(reward, 0)  # Phần thưởng dương khi đến đích
    
    def tearDown(self):
        """
        Dọn dẹp sau mỗi test case
        """
        self.env.close()

if __name__ == "__main__":
    unittest.main()