import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import gymnasium as gym
import pandas as pd
import json

from environments.urban_environment import UrbanEnvironment
from models.dqn_model import DQNAgent
from models.ppo_model import PPOAgent
from utils.visualization import save_training_video, plot_path
from utils.data_processing import create_simulation_data, preprocess_data, analyze_simulation_data

def run_simulation(model_type, model_path, num_episodes=10, render=True, save_video=True, log_dir='results', 
                 grid_size=100, num_obstacles=30, dynamic_obstacles=True, scenario='urban', device=None):
    """
    Chạy mô phỏng với mô hình đã huấn luyện.
    
    Args:
        model_type (str): Loại mô hình ('dqn' hoặc 'ppo')
        model_path (str): Đường dẫn đến mô hình
        num_episodes (int): Số lượng episode chạy
        render (bool): Có render môi trường không
        save_video (bool): Có lưu video không
        log_dir (str): Thư mục lưu kết quả
        grid_size (int): Kích thước lưới
        num_obstacles (int): Số lượng chướng ngại vật
        dynamic_obstacles (bool): Chướng ngại vật có di chuyển không
        scenario (str): Kịch bản ('urban', 'highway', 'parking', v.v.)
        device (str): Thiết bị sử dụng (cuda hoặc cpu)
    """
    # Tạo thư mục lưu kết quả
    os.makedirs(log_dir, exist_ok=True)
    
    # Xác định device (CPU hoặc GPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Khởi tạo môi trường dựa trên kịch bản
    render_mode = "rgb_array" if render else None
    
    if scenario == 'urban':
        env = UrbanEnvironment(
            grid_size=grid_size,
            num_obstacles=num_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            render_mode=render_mode
        )
    elif scenario == 'highway':
        # Có thể mở rộng với các môi trường khác trong tương lai
        env = UrbanEnvironment(
            grid_size=grid_size * 2,  # Đường cao tốc dài hơn
            num_obstacles=int(num_obstacles * 0.7),  # Ít chướng ngại vật hơn nhưng di chuyển nhanh
            dynamic_obstacles=True,
            render_mode=render_mode
        )
    elif scenario == 'parking':
        env = UrbanEnvironment(
            grid_size=grid_size // 2,  # Không gian nhỏ hơn
            num_obstacles=int(num_obstacles * 1.5),  # Nhiều chướng ngại vật hơn
            dynamic_obstacles=False,  # Chướng ngại vật tĩnh
            render_mode=render_mode
        )
    else:
        raise ValueError(f"Không hỗ trợ kịch bản: {scenario}")
    
    # Khởi tạo agent dựa trên loại mô hình
    if model_type == 'dqn':
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            device=device
        )
        
        # Tải mô hình
        if agent.load(model_path):
            print(f"Đã tải mô hình DQN từ {model_path}")
        else:
            print(f"Không thể tải mô hình DQN từ {model_path}")
            return
        
        # Chạy mô phỏng
        run_dqn_simulation(env, agent, num_episodes, render, save_video, log_dir, scenario)
    
    elif model_type == 'ppo':
        agent = PPOAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            device=device
        )
        
        # Tải mô hình
        actor_path = model_path.replace('.pt', '_actor.pt')
        critic_path = model_path.replace('.pt', '_critic.pt')
        
        if agent.load(actor_path, critic_path):
            print(f"Đã tải mô hình PPO từ {actor_path} và {critic_path}")
        else:
            print(f"Không thể tải mô hình PPO từ {actor_path} và {critic_path}")
            return
        
        # Chạy mô phỏng
        run_ppo_simulation(env, agent, num_episodes, render, save_video, log_dir, scenario)
    
    else:
        raise ValueError(f"Không hỗ trợ loại mô hình: {model_type}")
    
    # Đóng môi trường
    env.close()

def run_dqn_simulation(env, agent, num_episodes, render, save_video, log_dir, scenario):
    """
    Chạy mô phỏng với mô hình DQN.
    """
    # Lists để lưu kết quả
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    # Chạy các episodes
    for episode in tqdm(range(1, num_episodes + 1), desc=f"Mô phỏng {scenario} với DQN"):
        # Reset môi trường
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        terminated = False
        truncated = False
        
        # Buffer cho video nếu cần
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
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        
        # Kiểm tra xem có đạt đích không
        if np.linalg.norm(env.car_pos - env.target_pos) < 5:
            success_count += 1
        
        # Lưu kết quả episode
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        # Lưu video nếu cần
        if save_video and render and len(frames) > 0:
            try:
                video_path = os.path.join(log_dir, f"{scenario}_dqn_episode_{episode}.mp4")
                save_training_video(frames, video_path)
                
                # Vẽ và lưu đường đi
                path_plot_path = os.path.join(log_dir, f"{scenario}_dqn_path_{episode}.png")
                plot_path(
                    path, 
                    env.target_pos, 
                    env.obstacles, 
                    env.grid_size, 
                    path_plot_path,
                    title=f"{scenario.capitalize()} - Episode {episode} Path (DQN)"
                )
            except Exception as e:
                print(f"Lỗi khi lưu video hoặc đường đi: {e}")
    
    # Tính tỉ lệ thành công
    success_rate = (success_count / num_episodes) * 100
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    # In kết quả
    print(f"\nKết quả mô phỏng ({scenario} - DQN):")
    print(f"Tỉ lệ thành công: {success_rate:.2f}%")
    print(f"Phần thưởng trung bình: {avg_reward:.2f}")
    print(f"Số bước trung bình: {avg_steps:.2f}")
    
    # Lưu kết quả
    results = {
        'scenario': scenario,
        'model': 'DQN',
        'success_rate': success_rate,
        'avg_reward': float(avg_reward),
        'avg_steps': float(avg_steps),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_steps': [int(s) for s in episode_steps]
    }
    
    results_path = os.path.join(log_dir, f"{scenario}_dqn_simulation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Kết quả đã được lưu vào {results_path}")

def run_ppo_simulation(env, agent, num_episodes, render, save_video, log_dir, scenario):
    """
    Chạy mô phỏng với mô hình PPO.
    """
    # Lists để lưu kết quả
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    # Đặt mạng ở chế độ đánh giá
    agent.actor.eval()
    agent.critic.eval()
    
    # Chạy các episodes
    for episode in tqdm(range(1, num_episodes + 1), desc=f"Mô phỏng {scenario} với PPO"):
        # Reset môi trường
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        terminated = False
        truncated = False
        
        # Buffer cho video nếu cần
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
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        
        # Kiểm tra xem có đạt đích không
        if np.linalg.norm(env.car_pos - env.target_pos) < 5:
            success_count += 1
        
        # Lưu kết quả episode
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        # Lưu video nếu cần
        if save_video and render and len(frames) > 0:
            try:
                video_path = os.path.join(log_dir, f"{scenario}_ppo_episode_{episode}.mp4")
                save_training_video(frames, video_path)
                
                # Vẽ và lưu đường đi
                path_plot_path = os.path.join(log_dir, f"{scenario}_ppo_path_{episode}.png")
                plot_path(
                    path, 
                    env.target_pos, 
                    env.obstacles, 
                    env.grid_size, 
                    path_plot_path,
                    title=f"{scenario.capitalize()} - Episode {episode} Path (PPO)"
                )
            except Exception as e:
                print(f"Lỗi khi lưu video hoặc đường đi: {e}")
    
    # Tính tỉ lệ thành công
    success_rate = (success_count / num_episodes) * 100
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    # In kết quả
    print(f"\nKết quả mô phỏng ({scenario} - PPO):")
    print(f"Tỉ lệ thành công: {success_rate:.2f}%")
    print(f"Phần thưởng trung bình: {avg_reward:.2f}")
    print(f"Số bước trung bình: {avg_steps:.2f}")
    
    # Lưu kết quả
    results = {
        'scenario': scenario,
        'model': 'PPO',
        'success_rate': success_rate,
        'avg_reward': float(avg_reward),
        'avg_steps': float(avg_steps),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_steps': [int(s) for s in episode_steps]
    }
    
    results_path = os.path.join(log_dir, f"{scenario}_ppo_simulation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Kết quả đã được lưu vào {results_path}")

def create_data(num_samples=5000, grid_size=100, num_obstacles=30, noise_level=0.05, output_dir='data'):
    """
    Tạo dữ liệu mô phỏng.
    """
    print(f"Đang tạo {num_samples} mẫu dữ liệu mô phỏng...")
    data = create_simulation_data(
        num_samples=num_samples,
        grid_size=grid_size,
        num_obstacles=num_obstacles,
        noise_level=noise_level,
        output_dir=output_dir
    )
    
    # Tiền xử lý dữ liệu
    print("Đang tiền xử lý dữ liệu...")
    preprocessed_data = preprocess_data(data, output_dir=output_dir)
    
    # Phân tích dữ liệu
    print("Đang phân tích dữ liệu...")
    analyze_simulation_data(data, output_dir=output_dir)
    
    print(f"Hoàn thành tạo dữ liệu. Kết quả được lưu vào thư mục: {output_dir}")

def main():
    """
    Hàm chính
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Mô hình Tối ưu Hóa Đường đi Tự động cho Xe Tự hành')
    subparsers = parser.add_subparsers(dest='command', help='Lệnh')
    
    # Parser cho lệnh run
    run_parser = subparsers.add_parser('run', help='Chạy mô phỏng với mô hình đã huấn luyện')
    run_parser.add_argument('--model', type=str, default='dqn', choices=['dqn', 'ppo'], 
                           help='Loại mô hình (dqn hoặc ppo)')
    run_parser.add_argument('--model_path', type=str, required=True, 
                           help='Đường dẫn đến file mô hình')
    run_parser.add_argument('--episodes', type=int, default=10, 
                           help='Số lượng episode chạy')
    run_parser.add_argument('--render', type=bool, default=True, 
                           help='Có render môi trường không')
    run_parser.add_argument('--save_video', type=bool, default=True, 
                           help='Có lưu video không')
    run_parser.add_argument('--scenario', type=str, default='urban', 
                           choices=['urban', 'highway', 'parking'], 
                           help='Kịch bản mô phỏng')
    run_parser.add_argument('--grid_size', type=int, default=100, 
                           help='Kích thước lưới')
    run_parser.add_argument('--obstacles', type=int, default=30, 
                           help='Số lượng chướng ngại vật')
    run_parser.add_argument('--dynamic', type=bool, default=True, 
                           help='Chướng ngại vật có di chuyển không')
    run_parser.add_argument('--log_dir', type=str, default='results', 
                           help='Thư mục lưu kết quả')
    run_parser.add_argument('--device', type=str, default=None,
                           help='Thiết bị sử dụng (cuda hoặc cpu)')
    
    # Parser cho lệnh train
    train_parser = subparsers.add_parser('train', help='Huấn luyện mô hình mới')
    train_parser.add_argument('--model', type=str, default='dqn', choices=['dqn', 'ppo'],
                             help='Loại mô hình để huấn luyện (dqn hoặc ppo)')
    train_parser.add_argument('--episodes', type=int, default=500,
                             help='Số lượng episode huấn luyện')
    train_parser.add_argument('--render', type=bool, default=False,
                             help='Có render môi trường trong lúc huấn luyện không')
    train_parser.add_argument('--grid_size', type=int, default=100,
                             help='Kích thước lưới')
    train_parser.add_argument('--obstacles', type=int, default=30,
                             help='Số lượng chướng ngại vật')
    train_parser.add_argument('--dynamic', type=bool, default=True,
                             help='Chướng ngại vật có di chuyển không')
    train_parser.add_argument('--save_dir', type=str, default='models',
                             help='Thư mục lưu mô hình')
    train_parser.add_argument('--log_dir', type=str, default='results',
                             help='Thư mục lưu kết quả')
    train_parser.add_argument('--device', type=str, default=None,
                             help='Thiết bị sử dụng (cuda hoặc cpu)')
    
    # Parser cho lệnh evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Đánh giá mô hình')
    eval_parser.add_argument('--model', type=str, default='all', 
                            choices=['dqn', 'ppo', 'astar', 'all'],
                            help='Loại mô hình để đánh giá')
    eval_parser.add_argument('--model_path', type=str, default=None,
                            help='Đường dẫn đến mô hình (nếu không cung cấp, sẽ dùng mô hình tốt nhất)')
    eval_parser.add_argument('--episodes', type=int, default=100,
                            help='Số lượng episode đánh giá')
    eval_parser.add_argument('--render', type=bool, default=True,
                            help='Có render môi trường không')
    eval_parser.add_argument('--save_video', type=bool, default=True,
                            help='Có lưu video không')
    eval_parser.add_argument('--grid_size', type=int, default=100,
                            help='Kích thước lưới')
    eval_parser.add_argument('--obstacles', type=int, default=30,
                            help='Số lượng chướng ngại vật')
    eval_parser.add_argument('--dynamic', type=bool, default=True,
                            help='Chướng ngại vật có di chuyển không')
    eval_parser.add_argument('--log_dir', type=str, default='results',
                            help='Thư mục lưu kết quả')
    eval_parser.add_argument('--device', type=str, default=None,
                            help='Thiết bị sử dụng (cuda hoặc cpu)')
    
    # Parser cho lệnh create-data
    data_parser = subparsers.add_parser('create-data', help='Tạo dữ liệu mô phỏng')
    data_parser.add_argument('--samples', type=int, default=5000,
                            help='Số lượng mẫu dữ liệu')
    data_parser.add_argument('--grid_size', type=int, default=100,
                            help='Kích thước lưới')
    data_parser.add_argument('--obstacles', type=int, default=30,
                            help='Số lượng chướng ngại vật')
    data_parser.add_argument('--noise', type=float, default=0.05,
                            help='Mức độ nhiễu')
    data_parser.add_argument('--output_dir', type=str, default='data',
                            help='Thư mục đầu ra')
    
    args = parser.parse_args()
    
    # Xử lý các lệnh
    if args.command == 'run':
        run_simulation(
            model_type=args.model,
            model_path=args.model_path,
            num_episodes=args.episodes,
            render=args.render,
            save_video=args.save_video,
            log_dir=args.log_dir,
            grid_size=args.grid_size,
            num_obstacles=args.obstacles,
            dynamic_obstacles=args.dynamic,
            scenario=args.scenario,
            device=args.device
        )
    
    elif args.command == 'train':
        # Import ở đây để tránh import cycle
        from train import main as train_main
        
        # Tạo arguments cho train.py
        train_args = [
            '--model', args.model,
            '--episodes', str(args.episodes),
            '--render', str(args.render),
            '--grid_size', str(args.grid_size),
            '--obstacles', str(args.obstacles),
            '--dynamic', str(args.dynamic),
            '--save_dir', args.save_dir,
            '--log_dir', args.log_dir
        ]
        
        if args.device is not None:
            train_args.extend(['--device', args.device])
        
        # Gọi main của train.py
        train_main(train_args)
    
    elif args.command == 'evaluate':
        # Import ở đây để tránh import cycle
        from evaluate import main as eval_main
        
        # Tạo arguments cho evaluate.py
        eval_args = [
            '--model', args.model
        ]
        
        if args.model_path:
            eval_args.extend(['--model_path', args.model_path])
            
        eval_args.extend([
            '--episodes', str(args.episodes),
            '--render', str(args.render),
            '--save_video', str(args.save_video),
            '--grid_size', str(args.grid_size),
            '--obstacles', str(args.obstacles),
            '--dynamic', str(args.dynamic),
            '--log_dir', args.log_dir
        ])
        
        if args.device is not None:
            eval_args.extend(['--device', args.device])
        
        # Gọi main của evaluate.py
        eval_main(eval_args)
    
    elif args.command == 'create-data':
        create_data(
            num_samples=args.samples,
            grid_size=args.grid_size,
            num_obstacles=args.obstacles,
            noise_level=args.noise,
            output_dir=args.output_dir
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()