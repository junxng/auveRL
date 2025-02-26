import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import gymnasium as gym
import pandas as pd

# Import môi trường và mô hình
from environments.urban_environment import UrbanEnvironment
from models.dqn_model import DQNAgent
from models.ppo_model import PPOAgent
from utils.visualization import save_training_video, plot_path

def evaluate_dqn(env, agent, num_episodes, render=True, save_video=True, log_dir='results'):
    """
    Đánh giá mô hình DQN.
    """
    # Tạo thư mục log
    os.makedirs(log_dir, exist_ok=True)
    
    # Lists để lưu dữ liệu đánh giá
    episode_rewards = []
    episode_steps = []
    success_rate = 0
    collision_rate = 0
    timeout_rate = 0
    path_efficiency = []
    
    # Chạy đánh giá
    for episode in tqdm(range(1, num_episodes + 1), desc="Evaluating DQN"):
        # Reset môi trường
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        # Frame buffer cho render video (nếu cần)
        frames = []
        
        # Lưu đường đi
        path = [env.car_pos.copy()]
        
        # Vòng lặp một episode
        while not (terminated or truncated):
            # Chọn hành động
            action = agent.act(state, training=False)
            
            # Thực hiện hành động
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Cập nhật trạng thái
            state = next_state
            episode_reward += reward
            step += 1
            
            # Lưu đường đi
            path.append(env.car_pos.copy())
            
            # Render nếu cần
            if render:
                env.render_mode = "rgb_array"
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                env.render_mode = None
        
        # Xác định kết quả episode
        final_dist = np.linalg.norm(env.car_pos - env.target_pos)
        
        if final_dist < 5:  # Đến đích
            success_rate += 1
        elif truncated:  # Hết thời gian
            timeout_rate += 1
        else:  # Va chạm
            collision_rate += 1
        
        # Tính hiệu quả đường đi
        if len(path) > 1:
            path = np.array(path)
            actual_dist = 0
            for i in range(1, len(path)):
                actual_dist += np.linalg.norm(path[i] - path[i-1])
            
            straight_dist = np.linalg.norm(path[0] - env.target_pos)
            
            # Tỉ lệ hiệu quả (càng gần 1 càng tốt)
            if actual_dist > 0:
                efficiency = straight_dist / actual_dist
                path_efficiency.append(efficiency)
        
        # Lưu dữ liệu episode
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        # Lưu video và đường đi
        if save_video and render and len(frames) > 0:
            try:
                video_path = os.path.join(log_dir, f"eval_dqn_episode_{episode}.mp4")
                save_training_video(frames, video_path)
                
                # Vẽ và lưu đường đi
                path_plot_path = os.path.join(log_dir, f"eval_dqn_path_{episode}.png")
                plot_path(
                    path, 
                    env.target_pos, 
                    env.obstacles, 
                    env.grid_size, 
                    path_plot_path,
                    title=f"Episode {episode} Path"
                )
            except Exception as e:
                print(f"Lỗi khi lưu video hoặc đường đi: {e}")
    
    # Tính trung bình và độ lệch chuẩn
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_steps)
    std_steps = np.std(episode_steps)
    
    # Chuyển đổi tỉ lệ thành phần trăm
    success_rate = (success_rate / num_episodes) * 100
    collision_rate = (collision_rate / num_episodes) * 100
    timeout_rate = (timeout_rate / num_episodes) * 100
    
    avg_efficiency = np.mean(path_efficiency) if path_efficiency else 0
    std_efficiency = np.std(path_efficiency) if path_efficiency else 0
    
    # In kết quả
    print(f"\nEvaluation Results (DQN):")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Collision Rate: {collision_rate:.2f}%")
    print(f"Timeout Rate: {timeout_rate:.2f}%")
    print(f"Path Efficiency: {avg_efficiency:.3f} ± {std_efficiency:.3f}")
    
    # Lưu kết quả vào file CSV
    results = {
        'Algorithm': 'DQN',
        'Average Reward': avg_reward,
        'Std Reward': std_reward,
        'Average Steps': avg_steps,
        'Std Steps': std_steps,
        'Success Rate (%)': success_rate,
        'Collision Rate (%)': collision_rate,
        'Timeout Rate (%)': timeout_rate,
        'Path Efficiency': avg_efficiency,
        'Std Efficiency': std_efficiency
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(log_dir, 'dqn_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return results

def evaluate_ppo(env, agent, num_episodes, render=True, save_video=True, log_dir='results'):
    """
    Đánh giá mô hình PPO.
    """
    # Tạo thư mục log
    os.makedirs(log_dir, exist_ok=True)
    
    # Lists để lưu dữ liệu đánh giá
    episode_rewards = []
    episode_steps = []
    success_rate = 0
    collision_rate = 0
    timeout_rate = 0
    path_efficiency = []
    
    # Đặt mạng ở chế độ đánh giá
    agent.actor.eval()
    agent.critic.eval()
    
    # Chạy đánh giá
    for episode in tqdm(range(1, num_episodes + 1), desc="Evaluating PPO"):
        # Reset môi trường
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        # Frame buffer cho render video (nếu cần)
        frames = []
        
        # Lưu đường đi
        path = [env.car_pos.copy()]
        
        # Vòng lặp một episode
        while not (terminated or truncated):
            # Chọn hành động (deterministic)
            action, _, _ = agent.act(state, evaluation=True)
            
            # Thực hiện hành động
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Cập nhật trạng thái
            state = next_state
            episode_reward += reward
            step += 1
            
            # Lưu đường đi
            path.append(env.car_pos.copy())
            
            # Render nếu cần
            if render:
                env.render_mode = "rgb_array"
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                env.render_mode = None
        
        # Xác định kết quả episode
        final_dist = np.linalg.norm(env.car_pos - env.target_pos)
        
        if final_dist < 5:  # Đến đích
            success_rate += 1
        elif truncated:  # Hết thời gian
            timeout_rate += 1
        else:  # Va chạm
            collision_rate += 1
        
        # Tính hiệu quả đường đi
        if len(path) > 1:
            path = np.array(path)
            actual_dist = 0
            for i in range(1, len(path)):
                actual_dist += np.linalg.norm(path[i] - path[i-1])
            
            straight_dist = np.linalg.norm(path[0] - env.target_pos)
            
            # Tỉ lệ hiệu quả (càng gần 1 càng tốt)
            if actual_dist > 0:
                efficiency = straight_dist / actual_dist
                path_efficiency.append(efficiency)
        
        # Lưu dữ liệu episode
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        # Lưu video và đường đi
        if save_video and render and len(frames) > 0:
            try:
                video_path = os.path.join(log_dir, f"eval_ppo_episode_{episode}.mp4")
                save_training_video(frames, video_path)
                
                # Vẽ và lưu đường đi
                path_plot_path = os.path.join(log_dir, f"eval_ppo_path_{episode}.png")
                plot_path(
                    path, 
                    env.target_pos, 
                    env.obstacles, 
                    env.grid_size, 
                    path_plot_path,
                    title=f"Episode {episode} Path"
                )
            except Exception as e:
                print(f"Lỗi khi lưu video hoặc đường đi: {e}")
    
    # Tính trung bình và độ lệch chuẩn
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_steps)
    std_steps = np.std(episode_steps)
    
    # Chuyển đổi tỉ lệ thành phần trăm
    success_rate = (success_rate / num_episodes) * 100
    collision_rate = (collision_rate / num_episodes) * 100
    timeout_rate = (timeout_rate / num_episodes) * 100
    
    avg_efficiency = np.mean(path_efficiency) if path_efficiency else 0
    std_efficiency = np.std(path_efficiency) if path_efficiency else 0
    
    # In kết quả
    print(f"\nEvaluation Results (PPO):")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Collision Rate: {collision_rate:.2f}%")
    print(f"Timeout Rate: {timeout_rate:.2f}%")
    print(f"Path Efficiency: {avg_efficiency:.3f} ± {std_efficiency:.3f}")
    
    # Lưu kết quả vào file CSV
    results = {
        'Algorithm': 'PPO',
        'Average Reward': avg_reward,
        'Std Reward': std_reward,
        'Average Steps': avg_steps,
        'Std Steps': std_steps,
        'Success Rate (%)': success_rate,
        'Collision Rate (%)': collision_rate,
        'Timeout Rate (%)': timeout_rate,
        'Path Efficiency': avg_efficiency,
        'Std Efficiency': std_efficiency
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(log_dir, 'ppo_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return results

def evaluate_traditional(env, num_episodes, render=True, save_video=True, log_dir='results'):
    """
    Đánh giá thuật toán tìm đường truyền thống (A*) để so sánh.
    """
    # Tạo thư mục log
    os.makedirs(log_dir, exist_ok=True)
    
    # Lists để lưu dữ liệu đánh giá
    episode_rewards = []
    episode_steps = []
    success_rate = 0
    collision_rate = 0
    timeout_rate = 0
    path_efficiency = []
    
    # Chạy đánh giá
    for episode in tqdm(range(1, num_episodes + 1), desc="Evaluating A* Algorithm"):
        # Reset môi trường
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        # Frame buffer cho render video (nếu cần)
        frames = []
        
        # Lưu đường đi
        path = [env.car_pos.copy()]
        
        # Tìm đường đi với A*
        a_star_path = find_a_star_path(env)
        current_path_index = 0
        
        # Vòng lặp một episode
        while not (terminated or truncated) and current_path_index < len(a_star_path) - 1:
            # Lấy điểm tiếp theo trong đường đi
            next_point = a_star_path[current_path_index + 1]
            
            # Tính hướng cần di chuyển
            direction = next_point - env.car_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)  # Chuẩn hóa
            
            # Chọn hành động dựa trên hướng
            if direction[0] > 0.5 and direction[1] > 0.5:
                action = 0  # Tăng tốc (đi lên-phải)
            elif direction[0] > 0.5 and direction[1] < -0.5:
                action = 3  # Rẽ phải (đi xuống-phải)
            elif direction[0] < -0.5 and direction[1] > 0.5:
                action = 2  # Rẽ trái (đi lên-trái)
            elif direction[0] < -0.5 and direction[1] < -0.5:
                action = 1  # Phanh (đi xuống-trái)
            else:
                action = 4  # Giữ nguyên
            
            # Thực hiện hành động
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Cập nhật trạng thái
            state = next_state
            episode_reward += reward
            step += 1
            
            # Lưu đường đi
            path.append(env.car_pos.copy())
            
            # Kiểm tra xem đã đến điểm tiếp theo chưa
            if np.linalg.norm(env.car_pos - next_point) < 5:
                current_path_index += 1
            
            # Render nếu cần
            if render:
                env.render_mode = "rgb_array"
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                env.render_mode = None
        
        # Xác định kết quả episode
        final_dist = np.linalg.norm(env.car_pos - env.target_pos)
        
        if final_dist < 5:  # Đến đích
            success_rate += 1
        elif truncated:  # Hết thời gian
            timeout_rate += 1
        else:  # Va chạm
            collision_rate += 1
        
        # Tính hiệu quả đường đi
        if len(path) > 1:
            path = np.array(path)
            actual_dist = 0
            for i in range(1, len(path)):
                actual_dist += np.linalg.norm(path[i] - path[i-1])
            
            straight_dist = np.linalg.norm(path[0] - env.target_pos)
            
            # Tỉ lệ hiệu quả (càng gần 1 càng tốt)
            if actual_dist > 0:
                efficiency = straight_dist / actual_dist
                path_efficiency.append(efficiency)
        
        # Lưu dữ liệu episode
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        # Lưu video và đường đi
        if save_video and render and len(frames) > 0:
            try:
                video_path = os.path.join(log_dir, f"eval_astar_episode_{episode}.mp4")
                save_training_video(frames, video_path)
                
                # Vẽ và lưu đường đi
                path_plot_path = os.path.join(log_dir, f"eval_astar_path_{episode}.png")
                plot_path(
                    path, 
                    env.target_pos, 
                    env.obstacles, 
                    env.grid_size, 
                    path_plot_path,
                    title=f"Episode {episode} Path (A*)",
                    planned_path=a_star_path
                )
            except Exception as e:
                print(f"Lỗi khi lưu video hoặc đường đi: {e}")
    
    # Tính trung bình và độ lệch chuẩn
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_steps)
    std_steps = np.std(episode_steps)
    
    # Chuyển đổi tỉ lệ thành phần trăm
    success_rate = (success_rate / num_episodes) * 100
    collision_rate = (collision_rate / num_episodes) * 100
    timeout_rate = (timeout_rate / num_episodes) * 100
    
    avg_efficiency = np.mean(path_efficiency) if path_efficiency else 0
    std_efficiency = np.std(path_efficiency) if path_efficiency else 0
    
    # In kết quả
    print(f"\nEvaluation Results (A*):")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Collision Rate: {collision_rate:.2f}%")
    print(f"Timeout Rate: {timeout_rate:.2f}%")
    print(f"Path Efficiency: {avg_efficiency:.3f} ± {std_efficiency:.3f}")
    
    # Lưu kết quả vào file CSV
    results = {
        'Algorithm': 'A*',
        'Average Reward': avg_reward,
        'Std Reward': std_reward,
        'Average Steps': avg_steps,
        'Std Steps': std_steps,
        'Success Rate (%)': success_rate,
        'Collision Rate (%)': collision_rate,
        'Timeout Rate (%)': timeout_rate,
        'Path Efficiency': avg_efficiency,
        'Std Efficiency': std_efficiency
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(log_dir, 'astar_evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return results

def find_a_star_path(env):
    """
    Thuật toán A* đơn giản để tìm đường đi từ vị trí xe đến đích.
    """
    # Hàm heuristic (khoảng cách Euclidean đến đích)
    def heuristic(pos, target):
        return np.linalg.norm(np.array(pos) - np.array(target))
    
    # Rời rạc hóa không gian
    grid_resolution = 5  # Mỗi ô 5x5
    grid_size = env.grid_size // grid_resolution
    
    # Tạo grid biểu diễn môi trường
    grid = np.zeros((grid_size, grid_size))
    
    # Đánh dấu các chướng ngại vật trên grid
    for obs in env.obstacles:
        x, y = int(obs[0] / grid_resolution), int(obs[1] / grid_resolution)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[y, x] = 1  # Chướng ngại vật
            
            # Mở rộng chướng ngại vật (buffer)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        grid[ny, nx] = 1
    
    # Điểm bắt đầu và kết thúc
    start = (int(env.car_pos[0] / grid_resolution), int(env.car_pos[1] / grid_resolution))
    goal = (int(env.target_pos[0] / grid_resolution), int(env.target_pos[1] / grid_resolution))
    
    # Kiểm tra và điều chỉnh nếu điểm nằm trên chướng ngại vật
    if start[1] < grid_size and start[0] < grid_size and grid[start[1], start[0]] == 1:
        grid[start[1], start[0]] = 0
    if goal[1] < grid_size and goal[0] < grid_size and grid[goal[1], goal[0]] == 1:
        grid[goal[1], goal[0]] = 0
    
    # Thuật toán A*
    open_set = []
    closed_set = set()
    came_from = {}
    
    # g_score[n] là chi phí từ start đến n
    g_score = {start: 0}
    
    # f_score[n] = g_score[n] + heuristic(n)
    f_score = {start: heuristic(start, goal)}
    
    # Các bước di chuyển có thể (8 hướng)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    
    # Thêm điểm bắt đầu vào open_set
    open_set.append((f_score[start], start))
    
    while open_set:
        # Lấy node có f_score nhỏ nhất
        open_set.sort()
        current = open_set.pop(0)[1]
        
        # Nếu tìm thấy đích
        if current == goal:
            # Tái tạo đường đi
            path = []
            while current in came_from:
                path.append(np.array([current[0] * grid_resolution + grid_resolution / 2, 
                                    current[1] * grid_resolution + grid_resolution / 2]))
                current = came_from[current]
            
            # Thêm điểm bắt đầu và đảo ngược đường đi
            path.append(env.car_pos)
            path.reverse()
            
            # Thêm điểm đích
            path.append(env.target_pos)
            
            return path
        
        # Thêm vào closed_set
        closed_set.add(current)
        
        # Kiểm tra các node hàng xóm
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Kiểm tra biên
            if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                continue
            
            # Kiểm tra chướng ngại vật
            if grid[neighbor[1], neighbor[0]] == 1:
                continue
            
            # Kiểm tra closed_set
            if neighbor in closed_set:
                continue
            
            # Tính g_score cho neighbor
            tentative_g_score = g_score[current] + np.sqrt(dx**2 + dy**2)
            
            # Kiểm tra xem neighbor đã trong open_set chưa
            is_in_open_set = False
            for i, (_, node) in enumerate(open_set):
                if node == neighbor:
                    is_in_open_set = True
                    break
            
            # Nếu neighbor không trong open_set hoặc có g_score tốt hơn
            if not is_in_open_set or tentative_g_score < g_score.get(neighbor, float('inf')):
                # Cập nhật path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                # Thêm vào open_set nếu chưa
                if not is_in_open_set:
                    open_set.append((f_score[neighbor], neighbor))
    
    # Nếu không tìm thấy đường đi
    return [env.car_pos, env.target_pos]  # Đường thẳng trực tiếp

def compare_algorithms(results_dir='results'):
    """
    So sánh kết quả các thuật toán và tạo biểu đồ.
    """
    # Đọc kết quả từ các file CSV
    dqn_results_path = os.path.join(results_dir, 'dqn_evaluation_results.csv')
    ppo_results_path = os.path.join(results_dir, 'ppo_evaluation_results.csv')
    astar_results_path = os.path.join(results_dir, 'astar_evaluation_results.csv')
    
    try:
        dqn_results = pd.read_csv(dqn_results_path)
    except:
        dqn_results = None
        
    try:
        ppo_results = pd.read_csv(ppo_results_path)
    except:
        ppo_results = None
        
    try:
        astar_results = pd.read_csv(astar_results_path)
    except:
        astar_results = None
    
    # Kết hợp kết quả
    all_results = []
    if dqn_results is not None:
        all_results.append(dqn_results)
    if ppo_results is not None:
        all_results.append(ppo_results)
    if astar_results is not None:
        all_results.append(astar_results)
    
    if not all_results:
        print("Không tìm thấy kết quả đánh giá nào.")
        return
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Tạo biểu đồ so sánh
    plt.figure(figsize=(15, 10))
    
    # 1. Success Rate
    plt.subplot(2, 2, 1)
    bars = plt.bar(combined_results['Algorithm'], combined_results['Success Rate (%)'])
    plt.title('Success Rate (%)')
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Average Steps
    plt.subplot(2, 2, 2)
    bars = plt.bar(combined_results['Algorithm'], combined_results['Average Steps'])
    plt.title('Average Steps to Reach Goal')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 3. Path Efficiency
    plt.subplot(2, 2, 3)
    bars = plt.bar(combined_results['Algorithm'], combined_results['Path Efficiency'])
    plt.title('Path Efficiency (higher is better)')
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 4. Collision Rate
    plt.subplot(2, 2, 4)
    bars = plt.bar(combined_results['Algorithm'], combined_results['Collision Rate (%)'])
    plt.title('Collision Rate (%)')
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_path = os.path.join(results_dir, 'algorithm_comparison.png')
    plt.savefig(comparison_path)
    print(f"Comparison chart saved to {comparison_path}")
    
    # Tạo bảng so sánh chi tiết
    comparison_table = combined_results[['Algorithm', 'Success Rate (%)', 'Collision Rate (%)', 
                                        'Timeout Rate (%)', 'Average Steps', 'Path Efficiency']]
    comparison_csv = os.path.join(results_dir, 'algorithm_comparison.csv')
    comparison_table.to_csv(comparison_csv, index=False)
    print(f"Comparison table saved to {comparison_csv}")
    
    return comparison_table

def main(args=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate RL models for autonomous vehicles')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['dqn', 'ppo', 'astar', 'all'], 
                        help='Model to evaluate (dqn, ppo, astar, or all)')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=100, 
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', type=bool, default=True, 
                        help='Whether to render the environment')
    parser.add_argument('--save_video', type=bool, default=True, 
                        help='Whether to save videos')
    parser.add_argument('--log_dir', type=str, default='results', 
                        help='Directory to save logs and videos')
    parser.add_argument('--grid_size', type=int, default=100, 
                        help='Size of the grid environment')
    parser.add_argument('--obstacles', type=int, default=30, 
                        help='Number of obstacles in the environment')
    parser.add_argument('--dynamic', type=bool, default=True, 
                        help='Whether obstacles are dynamic or static')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (cuda or cpu)')
    
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    
    # Xác định device (CPU hoặc GPU)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Khởi tạo môi trường
    env = UrbanEnvironment(
        grid_size=args.grid_size,
        num_obstacles=args.obstacles,
        dynamic_obstacles=args.dynamic,
        render_mode="rgb_array" if args.render else None
    )
    
    # Đánh giá các mô hình
    if args.model in ['dqn', 'all']:
        # Khởi tạo DQN agent
        dqn_agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            device=device
        )
        
        # Tải mô hình
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join('models', 'dqn_model_best.pt')
        
        if dqn_agent.load(model_path):
            print(f"Loaded DQN model from {model_path}")
            # Đánh giá DQN
            evaluate_dqn(
                env=env,
                agent=dqn_agent,
                num_episodes=args.episodes,
                render=args.render,
                save_video=args.save_video,
                log_dir=args.log_dir
            )
        else:
            print(f"Failed to load DQN model from {model_path}")
    
    if args.model in ['ppo', 'all']:
        # Khởi tạo PPO agent
        ppo_agent = PPOAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            device=device
        )
        
        # Tải mô hình
        if args.model_path:
            actor_path = args.model_path.replace('.pt', '_actor.pt')
            critic_path = args.model_path.replace('.pt', '_critic.pt')
        else:
            actor_path = os.path.join('models', 'ppo_actor_best.pt')
            critic_path = os.path.join('models', 'ppo_critic_best.pt')
        
        if ppo_agent.load(actor_path, critic_path):
            print(f"Loaded PPO model from {actor_path} and {critic_path}")
            # Đánh giá PPO
            evaluate_ppo(
                env=env,
                agent=ppo_agent,
                num_episodes=args.episodes,
                render=args.render,
                save_video=args.save_video,
                log_dir=args.log_dir
            )
        else:
            print(f"Failed to load PPO model from {actor_path} and {critic_path}")
    
    if args.model in ['astar', 'all']:
        # Đánh giá thuật toán A*
        evaluate_traditional(
            env=env,
            num_episodes=args.episodes,
            render=args.render,
            save_video=args.save_video,
            log_dir=args.log_dir
        )
    
    # So sánh các thuật toán
    if args.model == 'all':
        compare_algorithms(results_dir=args.log_dir)
    
    # Đóng môi trường
    env.close()

if __name__ == "__main__":
    main()