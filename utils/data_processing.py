import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
import json

def create_simulation_data(num_samples=1000, grid_size=100, num_obstacles=30, noise_level=0.05, output_dir='data'):
    """
    Tạo dữ liệu mô phỏng cho việc học trước khi triển khai.
    
    Args:
        num_samples (int): Số lượng mẫu cần tạo
        grid_size (int): Kích thước lưới môi trường
        num_obstacles (int): Số lượng chướng ngại vật
        noise_level (float): Mức độ nhiễu thêm vào dữ liệu
        output_dir (str): Thư mục đầu ra
    
    Returns:
        dict: Dictionary chứa dữ liệu đã tạo
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo dữ liệu
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    # Tạo danh sách chướng ngại vật ngẫu nhiên
    obstacles = []
    for _ in range(num_obstacles):
        obstacle_pos = np.random.uniform(0, grid_size, size=2)
        obstacles.append(obstacle_pos)
    
    # Tạo dữ liệu mô phỏng
    for _ in range(num_samples):
        # Tạo trạng thái ngẫu nhiên
        car_pos = np.random.uniform(5, grid_size - 5, size=2)
        car_velocity = np.random.uniform(-3, 3, size=2)
        target_pos = np.random.uniform(5, grid_size - 5, size=2)
        
        # Tính cảm biến khoảng cách
        sensor_data = calculate_sensor_data(car_pos, obstacles, grid_size)
        
        # Kết hợp thành trạng thái hoàn chỉnh
        state = np.concatenate([car_pos, car_velocity, sensor_data, target_pos])
        
        # Chọn hành động ngẫu nhiên (nhưng có xu hướng hợp lý)
        action = choose_reasonable_action(car_pos, car_velocity, target_pos, obstacles, grid_size)
        
        # Tính toán trạng thái tiếp theo dựa trên mô phỏng đơn giản
        next_car_velocity = simulate_next_velocity(car_velocity, action)
        next_car_pos = car_pos + next_car_velocity
        
        # Kiểm tra kết thúc
        done = check_termination(next_car_pos, target_pos, obstacles, grid_size)
        
        # Tính toán phần thưởng
        reward = calculate_reward(car_pos, next_car_pos, target_pos, obstacles, done, grid_size)
        
        # Tính cảm biến khoảng cách mới
        next_sensor_data = calculate_sensor_data(next_car_pos, obstacles, grid_size)
        
        # Kết hợp thành trạng thái tiếp theo
        next_state = np.concatenate([next_car_pos, next_car_velocity, next_sensor_data, target_pos])
        
        # Thêm nhiễu nhỏ để tăng tính đa dạng
        state += np.random.normal(0, noise_level, state.shape)
        next_state += np.random.normal(0, noise_level, next_state.shape)
        
        # Lưu vào danh sách
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
    
    # Chuyển đổi thành mảng numpy
    data = {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_states': np.array(next_states),
        'dones': np.array(dones),
        'obstacles': np.array(obstacles)
    }
    
    # Lưu dữ liệu
    np.save(os.path.join(output_dir, 'simulation_data.npy'), data)
    
    # Lưu một phần dữ liệu dưới dạng CSV để dễ kiểm tra
    df = pd.DataFrame({
        'car_pos_x': [s[0] for s in states],
        'car_pos_y': [s[1] for s in states],
        'car_vel_x': [s[2] for s in states],
        'car_vel_y': [s[3] for s in states],
        'action': actions,
        'reward': rewards,
        'done': dones
    })
    df.to_csv(os.path.join(output_dir, 'simulation_data_sample.csv'), index=False)
    
    print(f"Created {num_samples} simulation data samples in {output_dir}")
    return data

def calculate_sensor_data(car_pos, obstacles, grid_size):
    """
    Tính toán dữ liệu cảm biến cho xe.
    
    Args:
        car_pos (array): Vị trí xe
        obstacles (list): Danh sách các chướng ngại vật
        grid_size (int): Kích thước lưới
    
    Returns:
        array: Dữ liệu cảm biến (8 hướng)
    """
    sensor_data = np.ones(8) * grid_size * 1.5  # Giá trị lớn ban đầu
    directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]
    
    for i, direction in enumerate(directions):
        dir_vector = np.array(direction) / np.linalg.norm(direction)
        
        for obs in obstacles:
            # Tính vector từ xe đến chướng ngại vật
            to_obstacle = obs - car_pos
            
            # Chiếu vector này lên vector hướng
            projection = np.dot(to_obstacle, dir_vector)
            
            # Chỉ xét chướng ngại vật ở phía trước theo hướng
            if projection > 0:
                # Tính khoảng cách từ chướng ngại vật đến đường thẳng theo hướng cảm biến
                perpendicular = to_obstacle - projection * dir_vector
                perp_distance = np.linalg.norm(perpendicular)
                
                # Nếu chướng ngại vật đủ gần đường thẳng và gần xe
                if perp_distance < 5 and projection < sensor_data[i]:
                    sensor_data[i] = projection
    
    return sensor_data

def choose_reasonable_action(car_pos, car_velocity, target_pos, obstacles, grid_size):
    """
    Chọn hành động hợp lý dựa trên heuristic đơn giản.
    
    Args:
        car_pos (array): Vị trí xe
        car_velocity (array): Vận tốc xe
        target_pos (array): Vị trí đích
        obstacles (list): Danh sách các chướng ngại vật
        grid_size (int): Kích thước lưới
    
    Returns:
        int: Hành động được chọn (0-4)
    """
    # Tính hướng đến đích
    to_target = target_pos - car_pos
    to_target_normalized = to_target / (np.linalg.norm(to_target) + 1e-8)
    
    # Kiểm tra chướng ngại vật gần nhất theo hướng đến đích
    min_obstacle_dist = grid_size * 2
    for obs in obstacles:
        dist = np.linalg.norm(obs - car_pos)
        if dist < min_obstacle_dist:
            min_obstacle_dist = dist
    
    # Nếu có chướng ngại vật gần, có 60% cơ hội tránh
    if min_obstacle_dist < 15 and np.random.random() < 0.6:
        # Chọn rẽ trái hoặc phải để tránh
        return np.random.choice([2, 3])
    
    # Nếu gần đích, có 70% cơ hội phanh
    if np.linalg.norm(to_target) < 10 and np.linalg.norm(car_velocity) > 1 and np.random.random() < 0.7:
        return 1  # Phanh
    
    # Nếu đang đi đúng hướng và vận tốc hợp lý, 60% cơ hội giữ nguyên
    current_direction = car_velocity / (np.linalg.norm(car_velocity) + 1e-8)
    if (np.dot(current_direction, to_target_normalized) > 0.8 and 
        1 < np.linalg.norm(car_velocity) < 4 and 
        np.random.random() < 0.6):
        return 4  # Giữ nguyên
    
    # Nếu đang đi chậm hoặc sai hướng, 70% cơ hội tăng tốc
    if (np.linalg.norm(car_velocity) < 1 or 
        np.dot(current_direction, to_target_normalized) < 0.3) and np.random.random() < 0.7:
        return 0  # Tăng tốc
    
    # Còn lại chọn ngẫu nhiên
    return np.random.randint(0, 5)

def simulate_next_velocity(car_velocity, action):
    """
    Mô phỏng vận tốc tiếp theo dựa trên hành động.
    
    Args:
        car_velocity (array): Vận tốc xe hiện tại
        action (int): Hành động (0-4)
    
    Returns:
        array: Vận tốc xe mới
    """
    next_velocity = car_velocity.copy()
    
    if action == 0:  # Tăng tốc
        next_velocity += np.array([0.5, 0.5])
    elif action == 1:  # Phanh
        next_velocity -= np.array([0.5, 0.5])
    elif action == 2:  # Rẽ trái
        next_velocity += np.array([-0.5, 0.5])
    elif action == 3:  # Rẽ phải
        next_velocity += np.array([0.5, -0.5])
    # action == 4 là giữ nguyên
    
    # Giới hạn vận tốc
    next_velocity = np.clip(next_velocity, -5, 5)
    
    return next_velocity

def check_termination(car_pos, target_pos, obstacles, grid_size):
    """
    Kiểm tra điều kiện kết thúc.
    
    Args:
        car_pos (array): Vị trí xe
        target_pos (array): Vị trí đích
        obstacles (list): Danh sách các chướng ngại vật
        grid_size (int): Kích thước lưới
    
    Returns:
        bool: True nếu đã đến đích hoặc va chạm
    """
    # Kiểm tra đã đến đích chưa
    if np.linalg.norm(car_pos - target_pos) < 5:
        return True
    
    # Kiểm tra va chạm với biên
    if (car_pos[0] < 0 or car_pos[0] >= grid_size or
        car_pos[1] < 0 or car_pos[1] >= grid_size):
        return True
    
    # Kiểm tra va chạm với chướng ngại vật
    for obs in obstacles:
        if np.linalg.norm(car_pos - obs) < 5:
            return True
    
    return False

def calculate_reward(car_pos, next_car_pos, target_pos, obstacles, done, grid_size):
    """
    Tính toán phần thưởng cho hành động.
    
    Args:
        car_pos (array): Vị trí xe hiện tại
        next_car_pos (array): Vị trí xe tiếp theo
        target_pos (array): Vị trí đích
        obstacles (list): Danh sách các chướng ngại vật
        done (bool): Trạng thái kết thúc
        grid_size (int): Kích thước lưới
    
    Returns:
        float: Phần thưởng
    """
    # Phần thưởng ban đầu
    reward = 0
    
    # Tính khoảng cách đến đích
    old_dist = np.linalg.norm(car_pos - target_pos)
    new_dist = np.linalg.norm(next_car_pos - target_pos)
    
    # Phần thưởng cho việc tiến gần đích
    reward += old_dist - new_dist
    
    # Kiểm tra xem đã đạt đích chưa
    if new_dist < 5:
        reward += 100  # Thưởng lớn cho việc đạt đích
        return reward
    
    # Phạt cho việc va chạm với biên
    if (next_car_pos[0] < 0 or next_car_pos[0] >= grid_size or
        next_car_pos[1] < 0 or next_car_pos[1] >= grid_size):
        reward -= 50
        return reward
    
    # Phạt cho việc va chạm với chướng ngại vật
    for obs in obstacles:
        if np.linalg.norm(next_car_pos - obs) < 5:
            reward -= 30
            return reward
    
    # Phạt nhẹ cho mỗi bước (khuyến khích hoàn thành nhanh)
    reward -= 0.1
    
    return reward

def preprocess_data(data, output_dir='data'):
    """
    Tiền xử lý dữ liệu cho việc huấn luyện.
    
    Args:
        data (dict): Dictionary chứa dữ liệu
        output_dir (str): Thư mục đầu ra
    
    Returns:
        dict: Dictionary chứa dữ liệu đã tiền xử lý
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuẩn hóa states và next_states
    scaler = MinMaxScaler()
    
    # Flatten states và next_states
    states_flat = data['states'].reshape(data['states'].shape[0], -1)
    next_states_flat = data['next_states'].reshape(data['next_states'].shape[0], -1)
    
    # Fit scaler trên cả states và next_states
    combined = np.vstack([states_flat, next_states_flat])
    scaler.fit(combined)
    
    # Transform
    states_normalized = scaler.transform(states_flat)
    next_states_normalized = scaler.transform(next_states_flat)
    
    # Reshape lại
    states_normalized = states_normalized.reshape(data['states'].shape)
    next_states_normalized = next_states_normalized.reshape(data['next_states'].shape)
    
    # Chuẩn hóa rewards
    rewards_mean = np.mean(data['rewards'])
    rewards_std = np.std(data['rewards']) + 1e-8  # Tránh chia cho 0
    rewards_normalized = (data['rewards'] - rewards_mean) / rewards_std
    
    # Tạo dữ liệu đã tiền xử lý
    preprocessed_data = {
        'states': states_normalized,
        'actions': data['actions'],
        'rewards': rewards_normalized,
        'next_states': next_states_normalized,
        'dones': data['dones'],
        'obstacles': data['obstacles']
    }
    
    # Lưu dữ liệu đã tiền xử lý
    np.save(os.path.join(output_dir, 'preprocessed_data.npy'), preprocessed_data)
    
    # Lưu scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Lưu thông tin chuẩn hóa
    normalization_info = {
        'rewards_mean': float(rewards_mean),
        'rewards_std': float(rewards_std)
    }
    
    with open(os.path.join(output_dir, 'normalization_info.json'), 'w') as f:
        json.dump(normalization_info, f)
    
    print(f"Preprocessed data saved to {output_dir}")
    return preprocessed_data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, output_dir='data'):
    """
    Chia dữ liệu thành tập train, validation và test.
    
    Args:
        data (dict): Dictionary chứa dữ liệu
        train_ratio (float): Tỉ lệ dữ liệu cho tập train
        val_ratio (float): Tỉ lệ dữ liệu cho tập validation
        test_ratio (float): Tỉ lệ dữ liệu cho tập test
        output_dir (str): Thư mục đầu ra
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # Kiểm tra tổng tỉ lệ
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Tổng các tỉ lệ phải bằng 1"
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy số lượng mẫu
    n_samples = len(data['states'])
    
    # Tạo indices và xáo trộn
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Chia indices
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Chia dữ liệu
    train_data = {k: v[train_indices] for k, v in data.items() if k != 'obstacles'}
    val_data = {k: v[val_indices] for k, v in data.items() if k != 'obstacles'}
    test_data = {k: v[test_indices] for k, v in data.items() if k != 'obstacles'}
    
    # Thêm obstacles vào các tập dữ liệu
    train_data['obstacles'] = data['obstacles']
    val_data['obstacles'] = data['obstacles']
    test_data['obstacles'] = data['obstacles']
    
    # Lưu dữ liệu
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    
    print(f"Data split into {len(train_indices)} train, {len(val_indices)} validation, and {len(test_indices)} test samples")
    return train_data, val_data, test_data

def analyze_simulation_data(data, output_dir='data'):
    """
    Phân tích dữ liệu mô phỏng.
    
    Args:
        data (dict): Dictionary chứa dữ liệu
        output_dir (str): Thư mục đầu ra
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Phân tích phân phối hành động
    actions = data['actions']
    action_counts = np.bincount(actions.astype(int))
    action_names = ['Tăng tốc', 'Phanh', 'Rẽ trái', 'Rẽ phải', 'Giữ nguyên']
    
    plt.figure(figsize=(10, 6))
    plt.bar(action_names, action_counts)
    plt.title('Phân phối hành động')
    plt.ylabel('Số lượng')
    plt.savefig(os.path.join(output_dir, 'action_distribution.png'))
    plt.close()
    
    # Phân tích phân phối phần thưởng
    rewards = data['rewards']
    
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=30)
    plt.title('Phân phối phần thưởng')
    plt.xlabel('Phần thưởng')
    plt.ylabel('Số lượng')
    plt.savefig(os.path.join(output_dir, 'reward_distribution.png'))
    plt.close()
    
    # Phân tích kết quả (done)
    dones = data['dones']
    done_counts = np.bincount(dones.astype(int))
    
    plt.figure(figsize=(8, 8))
    plt.pie(done_counts, labels=['Chưa hoàn thành', 'Hoàn thành'], autopct='%1.1f%%')
    plt.title('Tỉ lệ hoàn thành')
    plt.savefig(os.path.join(output_dir, 'completion_rate.png'))
    plt.close()
    
    # Phân tích vận tốc xe
    velocities = np.array([state[2:4] for state in data['states']])
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(velocity_magnitudes, bins=20)
    plt.title('Phân phối độ lớn vận tốc')
    plt.xlabel('Độ lớn vận tốc')
    plt.ylabel('Số lượng')
    plt.savefig(os.path.join(output_dir, 'velocity_distribution.png'))
    plt.close()
    
    # Tạo báo cáo tổng quát
    report = {
        'num_samples': len(data['states']),
        'action_distribution': {action_names[i]: int(count) for i, count in enumerate(action_counts)},
        'reward_stats': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards))
        },
        'completion_rate': float(np.mean(dones)),
        'velocity_stats': {
            'mean': float(np.mean(velocity_magnitudes)),
            'std': float(np.std(velocity_magnitudes)),
            'min': float(np.min(velocity_magnitudes)),
            'max': float(np.max(velocity_magnitudes))
        }
    }
    
    with open(os.path.join(output_dir, 'data_analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Data analysis completed and saved to {output_dir}")