import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class UrbanEnvironment(gym.Env):
    """
    Môi trường mô phỏng đô thị cho xe tự hành sử dụng Gymnasium.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=100, num_obstacles=30, dynamic_obstacles=True, render_mode=None, simple=False):
        super(UrbanEnvironment, self).__init__()
        
        # Thông số cấu hình
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.max_steps = grid_size * 2
        self.step_count = 0
        self.render_mode = render_mode
        self.simple = simple  # Chế độ đơn giản cho việc huấn luyện ban đầu
        self.collision_count = 0  # Đếm số lần va chạm
        self.timeout_count = 0    # Đếm số lần hết thời gian
        self.success_count = 0    # Đếm số lần thành công
        
        # Không gian quan sát: Vị trí xe (x, y), vận tốc (vx, vy), 
        # thông tin từ cảm biến (8 hướng), vị trí đích (x, y)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -5, -5] + [0] * 8 + [0, 0], dtype=np.float32),
            high=np.array([grid_size, grid_size, 5, 5] + [grid_size] * 8 + [grid_size, grid_size], dtype=np.float32),
            dtype=np.float32  # Chỉ định rõ kiểu dữ liệu để tránh warning
        )
        
        # Không gian hành động: [tăng tốc, phanh, rẽ trái, rẽ phải, giữ nguyên]
        self.action_space = spaces.Discrete(5)
        
        # Vị trí xe, vận tốc, và các chướng ngại vật
        self.car_pos = None
        self.car_velocity = None
        self.target_pos = None
        self.obstacles = None
        self.dynamic_obstacle_velocities = None
        
        # Biến cho việc render
        self.fig = None
        self.ax = None
        
        # Biến theo dõi
        self.last_distance_to_target = None
        self.initial_distance_to_target = None
        
        # Reset môi trường
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Khởi tạo lại môi trường với vị trí mới.
        """
        # Đặt seed nếu được cung cấp
        super().reset(seed=seed)
        
        self.step_count = 0
        
        # Vị trí xe ban đầu (góc dưới bên trái với jitter nhỏ)
        if self.simple:
            # Trong chế độ đơn giản, xe bắt đầu không quá xa mục tiêu
            self.car_pos = np.array([
                self.grid_size * 0.2 + self.np_random.uniform(-5, 5),
                self.grid_size * 0.2 + self.np_random.uniform(-5, 5)
            ])
        else:
            # Trong chế độ thường, xe bắt đầu ở góc
            self.car_pos = np.array([
                5.0 + self.np_random.uniform(-2, 2),
                5.0 + self.np_random.uniform(-2, 2)
            ])
        
        self.car_velocity = np.array([0.0, 0.0])
        
        # Vị trí đích (góc trên bên phải với jitter nhỏ)
        if self.simple:
            # Trong chế độ đơn giản, đích không quá xa
            self.target_pos = np.array([
                self.grid_size * 0.8 + self.np_random.uniform(-5, 5),
                self.grid_size * 0.8 + self.np_random.uniform(-5, 5)
            ])
        else:
            # Trong chế độ thường, đích ở góc đối diện
            self.target_pos = np.array([
                self.grid_size - 5.0 + self.np_random.uniform(-2, 2),
                self.grid_size - 5.0 + self.np_random.uniform(-2, 2)
            ])
        
        # Khởi tạo chướng ngại vật ngẫu nhiên
        self.obstacles = []
        
        # Giảm số lượng chướng ngại vật trong chế độ đơn giản
        num_obstacles = self.num_obstacles // 3 if self.simple else self.num_obstacles
        
        for _ in range(num_obstacles):
            # Tránh đặt chướng ngại vật quá gần xe hoặc đích
            while True:
                obstacle_pos = self.np_random.uniform(0, self.grid_size, size=2)
                min_distance = 15 if self.simple else 10  # Khoảng cách an toàn lớn hơn trong chế độ đơn giản
                
                if (np.linalg.norm(obstacle_pos - self.car_pos) > min_distance and 
                    np.linalg.norm(obstacle_pos - self.target_pos) > min_distance):
                    # Đảm bảo không chặn hoàn toàn đường đi trực tiếp trong chế độ đơn giản
                    if self.simple:
                        direct_path = self.target_pos - self.car_pos
                        direct_path_norm = direct_path / np.linalg.norm(direct_path)
                        projected_dist = np.abs(np.dot(obstacle_pos - self.car_pos, direct_path_norm))
                        perpendicular_dist = np.linalg.norm(
                            (obstacle_pos - self.car_pos) - 
                            projected_dist * direct_path_norm
                        )
                        
                        # Nếu chướng ngại vật quá gần đường thẳng giữa xe và đích, thử lại
                        if perpendicular_dist < 10 and 0 < projected_dist < np.linalg.norm(direct_path):
                            continue
                    
                    self.obstacles.append(obstacle_pos)
                    break
        
        # Khởi tạo vận tốc cho chướng ngại vật động
        if self.dynamic_obstacles:
            self.dynamic_obstacle_velocities = []
            # Vận tốc chậm hơn trong chế độ đơn giản
            max_velocity = 0.5 if self.simple else 1.0
            
            for _ in range(len(self.obstacles)):
                velocity = self.np_random.uniform(-max_velocity, max_velocity, size=2)
                self.dynamic_obstacle_velocities.append(velocity)
        
        # Lưu khoảng cách ban đầu đến đích để tính reward tương đối
        self.initial_distance_to_target = np.linalg.norm(self.car_pos - self.target_pos)
        self.last_distance_to_target = self.initial_distance_to_target
        
        # Trả về trạng thái ban đầu và thông tin
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Thực hiện hành động và trả về trạng thái mới, phần thưởng, kết thúc, bị cắt và thông tin bổ sung.
        """
        self.step_count += 1
        
        # Vận tốc cũ
        old_velocity = self.car_velocity.copy()
        old_position = self.car_pos.copy()
        
        # Xử lý hành động
        if action == 0:  # Tăng tốc
            self.car_velocity += np.array([0.5, 0.5])
        elif action == 1:  # Phanh
            self.car_velocity -= np.array([0.5, 0.5])
        elif action == 2:  # Rẽ trái
            self.car_velocity += np.array([-0.5, 0.5])
        elif action == 3:  # Rẽ phải
            self.car_velocity += np.array([0.5, -0.5])
        # action == 4 là giữ nguyên
        
        # Giới hạn vận tốc
        self.car_velocity = np.clip(self.car_velocity, -5, 5)
        
        # Vị trí xe mới
        new_car_pos = self.car_pos + self.car_velocity
        
        # Kiểm tra va chạm với biên
        if (new_car_pos[0] < 0 or new_car_pos[0] >= self.grid_size or
            new_car_pos[1] < 0 or new_car_pos[1] >= self.grid_size):
            terminated = True
            truncated = False
            reward = -50  # Giảm mức phạt so với trước đây
            self.collision_count += 1
            info = {"collision": True, "success": False, "timeout": False}
        else:
            # Cập nhật vị trí xe
            self.car_pos = new_car_pos
            
            # Di chuyển chướng ngại vật động
            if self.dynamic_obstacles:
                for i in range(len(self.obstacles)):
                    self.obstacles[i] += self.dynamic_obstacle_velocities[i]
                    
                    # Phản xạ khi chạm biên
                    if self.obstacles[i][0] <= 0 or self.obstacles[i][0] >= self.grid_size:
                        self.dynamic_obstacle_velocities[i][0] *= -1
                    if self.obstacles[i][1] <= 0 or self.obstacles[i][1] >= self.grid_size:
                        self.dynamic_obstacle_velocities[i][1] *= -1
            
            # Kiểm tra va chạm với chướng ngại vật
            collision = False
            for obs in self.obstacles:
                if np.linalg.norm(self.car_pos - obs) < 5:  # Khoảng cách va chạm
                    collision = True
                    break
            
            if collision:
                terminated = True
                truncated = False
                reward = -30  # Giảm mức phạt so với trước đây
                self.collision_count += 1
                info = {"collision": True, "success": False, "timeout": False}
            else:
                # Kiểm tra đã đến đích chưa
                dist_to_target = np.linalg.norm(self.car_pos - self.target_pos)
                
                if dist_to_target < 5:  # Đã đến đích
                    terminated = True
                    truncated = False
                    # Phần thưởng cho việc hoàn thành nhiệm vụ, càng nhanh càng tốt
                    reward = 200 - 0.5 * self.step_count
                    self.success_count += 1
                    info = {"collision": False, "success": True, "timeout": False}
                else:
                    terminated = False
                    truncated = self.step_count >= self.max_steps
                    
                    if truncated:
                        self.timeout_count += 1
                        info = {"collision": False, "success": False, "timeout": True}
                    else:
                        info = {"collision": False, "success": False, "timeout": False}
                    
                    # Phần thưởng dựa trên sự tiến bộ hướng về đích
                    # Sử dụng khoảng cách tương đối so với ban đầu
                    old_dist = self.last_distance_to_target
                    new_dist = dist_to_target
                    self.last_distance_to_target = new_dist
                    
                    # Tăng mức thưởng cho việc tiến gần đích
                    progress_reward = 2.0 * (old_dist - new_dist)
                    
                    # Thưởng nhỏ cho việc duy trì vận tốc ổn định và hướng về phía đích
                    speed_reward = 0
                    direction_reward = 0
                    
                    if np.linalg.norm(self.car_velocity) > 0.1:
                        speed_reward = 0.2
                        
                        # Tính hướng di chuyển
                        normalized_velocity = self.car_velocity / np.linalg.norm(self.car_velocity)
                        to_target = self.target_pos - self.car_pos
                        if np.linalg.norm(to_target) > 0:
                            normalized_target = to_target / np.linalg.norm(to_target)
                            direction_alignment = np.dot(normalized_velocity, normalized_target)
                            direction_reward = 0.3 * max(0, direction_alignment)
                    
                    # Giảm phạt cho mỗi bước để khuyến khích agent khám phá
                    step_penalty = -0.05
                    
                    # Tổng reward
                    reward = progress_reward + speed_reward + direction_reward + step_penalty
                    
                    # Thêm thông tin vào info
                    info.update({
                        "distance_to_target": dist_to_target,
                        "progress_reward": progress_reward,
                        "speed_reward": speed_reward,
                        "direction_reward": direction_reward,
                        "step_penalty": step_penalty,
                        "normalized_distance": dist_to_target / self.initial_distance_to_target
                    })
        
        # Render môi trường nếu cần
        if self.render_mode == "human":
            self.render()
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Trả về trạng thái hiện tại của môi trường.
        """
        # Cảm biến khoảng cách theo 8 hướng
        sensor_data = self._get_sensor_data()
        
        # Kết hợp tất cả thông tin trạng thái
        state = np.concatenate([
            self.car_pos,            # Vị trí xe (2)
            self.car_velocity,       # Vận tốc xe (2)
            sensor_data,             # Dữ liệu cảm biến (8)
            self.target_pos          # Vị trí đích (2)
        ])
        
        return state
    
    def _get_sensor_data(self):
        """
        Mô phỏng cảm biến khoảng cách theo 8 hướng.
        """
        sensor_data = np.zeros(8)
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        for i, direction in enumerate(directions):
            dir_vector = np.array(direction) / np.linalg.norm(direction)
            
            # Tìm chướng ngại vật gần nhất theo hướng này
            min_distance = self.grid_size * 1.5  # Giá trị lớn ban đầu
            
            for obs in self.obstacles:
                # Tính vector từ xe đến chướng ngại vật
                to_obstacle = obs - self.car_pos
                
                # Chiếu vector này lên vector hướng
                projection = np.dot(to_obstacle, dir_vector)
                
                # Chỉ xét chướng ngại vật ở phía trước theo hướng
                if projection > 0:
                    # Tính khoảng cách từ chướng ngại vật đến đường thẳng theo hướng cảm biến
                    perpendicular = to_obstacle - projection * dir_vector
                    perp_distance = np.linalg.norm(perpendicular)
                    
                    # Nếu chướng ngại vật đủ gần đường thẳng và gần xe
                    if perp_distance < 5 and projection < min_distance:
                        min_distance = projection
            
            sensor_data[i] = min_distance
        
        return sensor_data
    
    def render(self):
        """
        Hiển thị môi trường.
        """
        if self.render_mode is None:
            return
            
        if self.fig is None:
            plt.switch_backend('agg')  # Chuyển sang backend agg cho rendering
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
    
        self.ax.clear()
    
        # Vẽ biên
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
    
        # Vẽ xe (hình tròn màu xanh)
        car = Circle((self.car_pos[0], self.car_pos[1]), 2, color='blue')
        self.ax.add_patch(car)
    
        # Vẽ mũi tên chỉ hướng vận tốc
        if np.linalg.norm(self.car_velocity) > 0.1:
            velocity_dir = self.car_velocity / np.linalg.norm(self.car_velocity) * 5
            self.ax.arrow(self.car_pos[0], self.car_pos[1], 
                    velocity_dir[0], velocity_dir[1], 
                    head_width=2, head_length=2, fc='blue', ec='blue')
    
        # Vẽ đích (hình tròn màu xanh lá)
        target = Circle((self.target_pos[0], self.target_pos[1]), 3, color='green')
        self.ax.add_patch(target)
    
        # Vẽ chướng ngại vật (hình tròn màu đỏ)
        for obs in self.obstacles:
            obstacle = Circle((obs[0], obs[1]), 3, color='red')
            self.ax.add_patch(obstacle)
    
        # Vẽ cảm biến
        sensor_data = self._get_sensor_data()
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
    
        for i, direction in enumerate(directions):
            dir_vector = np.array(direction) / np.linalg.norm(direction)
            end_point = self.car_pos + dir_vector * sensor_data[i]
            self.ax.plot([self.car_pos[0], end_point[0]], 
                    [self.car_pos[1], end_point[1]], 'y--', alpha=0.3)
    
        # Hiển thị thông tin
        info_text = f"Step: {self.step_count}\nVelocity: [{self.car_velocity[0]:.1f}, {self.car_velocity[1]:.1f}]\n"
        info_text += f"Distance: {np.linalg.norm(self.car_pos - self.target_pos):.1f}"
        self.ax.text(5, self.grid_size - 10, info_text, bbox=dict(facecolor='white', alpha=0.5))
    
        self.fig.canvas.draw()
    
        if self.render_mode == "human":
            plt.pause(0.1)
            return None
        elif self.render_mode == "rgb_array":
            # Chuyển đổi canvas thành mảng RGB - sử dụng cách tương thích với nhiều backend
            try:
                # Phương pháp 1: Sử dụng với backend Agg
                img = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
                img = img[:,:,:3]  # Chỉ lấy kênh RGB, bỏ qua kênh Alpha
                return img
            except Exception as e1:
                try:
                    # Phương pháp 2: Thử dùng tostring_argb nếu tostring_rgb không có
                    buf = self.fig.canvas.tostring_argb()
                    width, height = self.fig.canvas.get_width_height()
                    img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
                
                    # Chuyển từ ARGB sang RGB
                    return img[:, :, 1:4]
                except Exception as e2:
                    print(f"Không thể render hình ảnh: {e1}, sau đó: {e2}")
                    return None
                
    def close(self):
        """
        Đóng môi trường.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            
    def get_stats(self):
        """
        Trả về thống kê về môi trường.
        """
        total_episodes = self.success_count + self.collision_count + self.timeout_count
        if total_episodes == 0:
            total_episodes = 1  # Tránh chia cho 0
        
        return {
            "success_count": self.success_count,
            "collision_count": self.collision_count,
            "timeout_count": self.timeout_count,
            "success_rate": self.success_count / total_episodes * 100,
            "collision_rate": self.collision_count / total_episodes * 100,
            "timeout_rate": self.timeout_count / total_episodes * 100
        }