import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import gymnasium as gym

# Import môi trường và mô hình
from environments.urban_environment import UrbanEnvironment
from models.dqn_model import DQNAgent
from models.ppo_model import PPOAgent
from utils.visualization import plot_learning_curve, save_training_video

def train_dqn(env, agent, num_episodes, render=False, save_freq=100, log_freq=10, save_dir='models', log_dir='results'):
    """
    Huấn luyện mô hình DQN.
    """
    # Tạo thư mục lưu mô hình và log
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Lists để lưu dữ liệu huấn luyện
    episode_rewards = []
    episode_steps = []
    epsilon_history = []
    loss_history = []
    
    # Training loop
    best_average = -np.inf
    
    for episode in tqdm(range(1, num_episodes + 1), desc="Training DQN"):
        # Reset môi trường
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        episode_loss = []
        
        # Frame buffer cho render video (nếu cần)
        frames = []
        
        while not (terminated or truncated):
            # Chọn hành động
            action = agent.act(state)
            
            # Thực hiện hành động
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Lưu vào bộ nhớ
            agent.remember(state, action, reward, next_state, terminated)
            
            # Cập nhật trạng thái
            state = next_state
            episode_reward += reward
            step += 1
            
            # Huấn luyện mô hình
            loss = agent.replay()
            if loss is not None:
                episode_loss.append(loss)
            
            # Render nếu cần
            if render and episode % log_freq == 0:
                try:
                    env.render_mode = "rgb_array"
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    env.render_mode = None
                except Exception as e:
                    print(f"Lỗi khi render: {e}")
                    render = False
        
        # Lưu dữ liệu episode
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        epsilon_history.append(agent.epsilon)
        
        # Lưu loss trung bình nếu có
        if episode_loss:
            loss_history.append(np.mean(episode_loss))
        
        # Log progress
        if episode % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_steps = np.mean(episode_steps[-log_freq:])
            avg_loss = np.mean(loss_history[-log_freq:]) if loss_history else "N/A"
            print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Epsilon: {agent.epsilon:.4f}, Loss: {avg_loss}")
            
            # Lưu video
            if render and len(frames) > 0:
                try:
                    video_path = os.path.join(log_dir, f"dqn_episode_{episode}.mp4")
                    save_training_video(frames, video_path)
                except Exception as e:
                    print(f"Không thể lưu video: {e}")
        
        # Lưu mô hình tốt nhất và định kỳ
        if episode % save_freq == 0:
            model_path = os.path.join(save_dir, f"dqn_model_episode_{episode}.pt")
            agent.save(model_path)
            print(f"Model saved to {model_path}")
        
        # Lưu mô hình tốt nhất
        if episode >= log_freq:
            current_avg = np.mean(episode_rewards[-log_freq:])
            if current_avg > best_average:
                best_average = current_avg
                best_model_path = os.path.join(save_dir, "dqn_model_best.pt")
                agent.save(best_model_path)
                print(f"Best model saved with average reward: {best_average:.2f}")
    
    # Lưu mô hình cuối cùng
    final_model_path = os.path.join(save_dir, "dqn_model_final.pt")
    agent.save(final_model_path)
    
    # Trực quan hóa quá trình huấn luyện
    plot_learning_curve(
        episode_rewards, 
        os.path.join(log_dir, 'dqn_rewards.png'), 
        title="DQN Learning Curve", 
        x_label="Episode", 
        y_label="Reward",
        window=log_freq
    )
    
    plot_learning_curve(
        epsilon_history, 
        os.path.join(log_dir, 'dqn_epsilon.png'), 
        title="DQN Epsilon Decay", 
        x_label="Episode", 
        y_label="Epsilon",
        window=1
    )
    
    # Vẽ biểu đồ loss nếu có dữ liệu
    if loss_history:
        plot_learning_curve(
            loss_history, 
            os.path.join(log_dir, 'dqn_loss.png'), 
            title="DQN Loss", 
            x_label="Episode", 
            y_label="Loss",
            window=log_freq
        )
    
    return episode_rewards, episode_steps

def train_ppo(env, agent, num_episodes, render=False, save_freq=100, log_freq=10, save_dir='models', log_dir='results'):
    """
    Huấn luyện mô hình PPO.
    """
    # Tạo thư mục lưu mô hình và log
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Lists để lưu dữ liệu huấn luyện
    episode_rewards = []
    episode_steps = []
    
    # Training loop
    best_average = -np.inf
    
    for episode in tqdm(range(1, num_episodes + 1), desc="Training PPO"):
        # Reset môi trường
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        # Frame buffer cho render video (nếu cần)
        frames = []
        
        # Vòng lặp một episode
        while not (terminated or truncated):
            # Chọn hành động theo policy hiện tại
            action, value, log_prob = agent.act(state)
            
            # Thực hiện hành động
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Lưu vào bộ nhớ
            agent.remember(state, action, reward, value, log_prob, terminated)
            
            # Cập nhật trạng thái
            state = next_state
            episode_reward += reward
            step += 1
            
            # Render nếu cần
            if render and episode % log_freq == 0:
                try:
                    env.render_mode = "rgb_array"
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    env.render_mode = None
                except Exception as e:
                    print(f"Lỗi khi render: {e}")
                    render = False
        
        # Huấn luyện sau mỗi episode
        agent.train()
        
        # Lưu dữ liệu episode
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        # Log progress
        if episode % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_steps = np.mean(episode_steps[-log_freq:])
            print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}")
            
            # Lưu video
            if render and len(frames) > 0:
                try:
                    video_path = os.path.join(log_dir, f"ppo_episode_{episode}.mp4")
                    save_training_video(frames, video_path)
                except Exception as e:
                    print(f"Không thể lưu video: {e}")
        
        # Lưu mô hình tốt nhất và định kỳ
        if episode % save_freq == 0:
            actor_path = os.path.join(save_dir, f"ppo_actor_episode_{episode}.pt")
            critic_path = os.path.join(save_dir, f"ppo_critic_episode_{episode}.pt")
            agent.save(actor_path, critic_path)
            print(f"Model saved to {actor_path} and {critic_path}")
        
        # Lưu mô hình tốt nhất
        if episode >= log_freq:
            current_avg = np.mean(episode_rewards[-log_freq:])
            if current_avg > best_average:
                best_average = current_avg
                best_actor_path = os.path.join(save_dir, "ppo_actor_best.pt")
                best_critic_path = os.path.join(save_dir, "ppo_critic_best.pt")
                agent.save(best_actor_path, best_critic_path)
                print(f"Best model saved with average reward: {best_average:.2f}")
    
    # Lưu mô hình cuối cùng
    final_actor_path = os.path.join(save_dir, "ppo_actor_final.pt")
    final_critic_path = os.path.join(save_dir, "ppo_critic_final.pt")
    agent.save(final_actor_path, final_critic_path)
    
    # Trực quan hóa quá trình huấn luyện
    plot_learning_curve(
        episode_rewards, 
        os.path.join(log_dir, 'ppo_rewards.png'), 
        title="PPO Learning Curve", 
        x_label="Episode", 
        y_label="Reward",
        window=log_freq
    )
    
    return episode_rewards, episode_steps

def main(args=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train RL models for autonomous vehicles')
    parser.add_argument('--model', type=str, default='dqn', choices=['dqn', 'ppo'], 
                        help='RL algorithm to use (dqn or ppo)')
    parser.add_argument('--episodes', type=int, default=500, 
                        help='Number of episodes to train')
    parser.add_argument('--render', type=bool, default=False, 
                        help='Whether to render the environment')
    parser.add_argument('--save_freq', type=int, default=100, 
                        help='Frequency to save the model')
    parser.add_argument('--log_freq', type=int, default=10, 
                        help='Frequency to log progress')
    parser.add_argument('--save_dir', type=str, default='models', 
                        help='Directory to save models')
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
    
    # Khởi tạo agent dựa trên thuật toán được chọn
    if args.model == 'dqn':
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            memory_size=10000,
            batch_size=64,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            device=device
        )
        
        # Huấn luyện DQN
        rewards, steps = train_dqn(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            render=args.render,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
    else:  # ppo
        agent = PPOAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            gamma=0.99,
            clip_ratio=0.2,
            policy_lr=0.0003,
            value_lr=0.001,
            batch_size=64,
            epochs=10,
            lmbda=0.95,
            device=device
        )
        
        # Huấn luyện PPO
        rewards, steps = train_ppo(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            render=args.render,
            save_freq=args.save_freq,
            log_freq=args.log_freq,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
    
    # Đóng môi trường
    env.close()
    
    print(f"Training completed. Final average reward (last 10 episodes): {np.mean(rewards[-10:]):.2f}")

if __name__ == "__main__":
    main()