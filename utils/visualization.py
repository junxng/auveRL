import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation

def plot_learning_curve(data, filename, title="Learning Curve", x_label="Episodes", y_label="Rewards", window=10):
    """
    Vẽ và lưu đường cong học tập.
    
    Args:
        data (list): Dữ liệu cần vẽ
        filename (str): Đường dẫn file lưu hình ảnh
        title (str): Tiêu đề biểu đồ
        x_label (str): Nhãn trục x
        y_label (str): Nhãn trục y
        window (int): Kích thước cửa sổ cho đường trung bình động
    """
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Tạo figure
    plt.figure(figsize=(10, 6))
    
    # Vẽ dữ liệu gốc (mờ)
    plt.plot(data, alpha=0.3, color='blue')
    
    # Vẽ đường trung bình động (smoothed)
    if len(data) >= window:
        smoothed_data = []
        for i in range(len(data) - window + 1):
            smoothed_data.append(np.mean(data[i:i+window]))
        
        # Vẽ đường smoothed
        plt.plot(range(window-1, len(data)), smoothed_data, color='red', linewidth=2)
    
    # Thêm tiêu đề và nhãn
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Lưu hình ảnh
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_path(path, target, obstacles, grid_size, filename, title="Path", planned_path=None):
    """
    Vẽ và lưu đường đi của xe.
    
    Args:
        path (list or array): Danh sách các điểm tạo thành đường đi
        target (array): Vị trí đích
        obstacles (list): Danh sách các chướng ngại vật
        grid_size (int): Kích thước lưới
        filename (str): Đường dẫn file lưu hình ảnh
        title (str): Tiêu đề biểu đồ
        planned_path (list, optional): Đường đi đã được lên kế hoạch trước (nếu có)
    """
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Tạo figure
    plt.figure(figsize=(10, 10))
    
    # Vẽ biên
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    
    # Vẽ chướng ngại vật
    for obs in obstacles:
        obstacle = Circle((obs[0], obs[1]), 3, color='red', alpha=0.7)
        plt.gca().add_patch(obstacle)
    
    # Vẽ đường đi đã lên kế hoạch (nếu có)
    if planned_path is not None:
        planned_path = np.array(planned_path)
        plt.plot(planned_path[:, 0], planned_path[:, 1], 'y--', alpha=0.7, linewidth=2, label='Planned Path')
    
    # Vẽ đường đi thực tế
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Actual Path')
    
    # Vẽ điểm bắt đầu
    plt.scatter(path[0, 0], path[0, 1], color='blue', s=100, marker='o', label='Start')
    
    # Vẽ điểm đích
    plt.scatter(target[0], target[1], color='green', s=100, marker='*', label='Target')
    
    # Thêm tiêu đề và legend
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Lưu hình ảnh
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def save_training_frames(frames, filename_prefix, log_dir):
    """
    Lưu các frames riêng lẻ thay vì video.
    
    Args:
        frames (list): Danh sách các frames
        filename_prefix (str): Tiền tố cho tên file
        log_dir (str): Thư mục lưu frames
    """
    if not frames:
        print(f"Không có frames để lưu")
        return
    
    # Tạo thư mục lưu nếu chưa tồn tại
    frames_dir = os.path.join(log_dir, f"{filename_prefix}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Lưu tất cả các frames
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        plt.imsave(frame_path, frame)
    
    # Lưu frame đầu và frame cuối riêng để dễ kiểm tra
    first_frame_path = os.path.join(log_dir, f"{filename_prefix}_first.png")
    last_frame_path = os.path.join(log_dir, f"{filename_prefix}_last.png")
    
    plt.imsave(first_frame_path, frames[0])
    plt.imsave(last_frame_path, frames[-1])
    
    print(f"Đã lưu {len(frames)} frames tại: {frames_dir}")
    print(f"Đã lưu frames đầu và cuối tại: {first_frame_path} và {last_frame_path}")
    print(f"Frames có giống nhau không: {np.array_equal(frames[0], frames[-1])}")

def save_training_video(frames, filename, fps=10):
    """
    Lưu video từ các frames.
    
    Args:
        frames (list): Danh sách các frames
        filename (str): Đường dẫn file lưu video
        fps (int): Frames per second
    """
    if not frames:
        print(f"Không có frames để lưu thành video: {filename}")
        return
        
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    try:
        # Thử sử dụng thư viện av trực tiếp
        try:
            import av
            container = av.open(filename, mode='w')
            stream = container.add_stream('h264', rate=fps)
            stream.width = frames[0].shape[1]
            stream.height = frames[0].shape[0]
            stream.pix_fmt = 'yuv420p'
            
            for frame in frames:
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(frame)
                container.mux(packet)
            
            # Flush remaining packets
            packet = stream.encode(None)
            container.mux(packet)
            container.close()
            print(f"Đã lưu video tại: {filename}")
            
        except (ImportError, Exception) as e:
            # Nếu không thể sử dụng av trực tiếp, thử dùng imageio không có tham số phức tạp
            try:
                # Sử dụng imageio với các tham số cơ bản
                imageio.mimsave(filename, frames, fps=fps)
                print(f"Đã lưu video tại: {filename}")
            except Exception as e2:
                # Thử lưu dưới dạng GIF nếu MP4 không thành công
                gif_filename = filename.replace('.mp4', '.gif')
                print(f"Không thể lưu MP4: {str(e2)}. Đang lưu dưới dạng GIF: {gif_filename}")
                try:
                    imageio.mimsave(gif_filename, frames, fps=fps)
                    print(f"Đã lưu GIF tại: {gif_filename}")
                except Exception as e3:
                    print(f"Không thể lưu GIF: {str(e3)}")
                    raise e3
            
    except Exception as e:
        print(f"Lỗi khi lưu video: {str(e)}")
        
        # Lưu dự phòng: Lưu frame đầu tiên và cuối cùng dưới dạng PNG
        try:
            if frames:
                first_frame_path = os.path.join(os.path.dirname(filename), f"{os.path.basename(filename).split('.')[0]}_first.png")
                last_frame_path = os.path.join(os.path.dirname(filename), f"{os.path.basename(filename).split('.')[0]}_last.png")
                
                plt.imsave(first_frame_path, frames[0])
                plt.imsave(last_frame_path, frames[-1])
                print(f"Đã lưu frames đầu và cuối tại: {first_frame_path} và {last_frame_path}")
                print(f"Frames có giống nhau không: {np.array_equal(frames[0], frames[-1])}")
        except Exception as e2:
            print(f"Không thể lưu frames riêng lẻ: {str(e2)}")

def debug_render(env, actions=None, filename="debug_frames.png"):
    """
    Kiểm tra chức năng render của môi trường.
    
    Args:
        env: Môi trường cần kiểm tra
        actions: Danh sách các hành động để thực hiện
        filename: Tên file để lưu debug frames
    """
    # Lưu render_mode hiện tại để khôi phục sau
    original_render_mode = env.render_mode
    env.render_mode = "rgb_array"
    
    # Reset môi trường
    state, _ = env.reset()
    
    frames = []
    states = []
    rewards = []
    positions = []
    
    # Lưu trạng thái ban đầu
    frame = env.render()
    if frame is not None:
        frames.append(frame.copy())  # Cần .copy() để tránh tham chiếu
        positions.append(env.car_pos.copy())
        states.append(state)
    
    # Nếu không có hành động được cung cấp, sử dụng một số hành động mặc định
    if actions is None:
        actions = [0, 2, 3, 0, 1, 4]  # Tăng tốc, rẽ trái, rẽ phải, tăng tốc, phanh, giữ nguyên
    
    # Thực hiện các hành động
    for action in actions:
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        states.append(next_state)
        positions.append(env.car_pos.copy())
        
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())
        
        if terminated or truncated:
            break
    
    # Khôi phục render_mode
    env.render_mode = original_render_mode
    
    # In thông tin debug
    print(f"\n=== Debug Information ===")
    print(f"Number of frames: {len(frames)}")
    print(f"Actions taken: {actions[:len(rewards)]}")
    print(f"Rewards: {rewards}")
    print(f"Positions: {positions}")
    print(f"First and last position same: {np.array_equal(positions[0], positions[-1])}")
    
    # Lưu frames để kiểm tra
    if len(frames) > 1:
        plt.figure(figsize=(15, 10))
        
        # Nếu có nhiều hơn 2 frames, hiển thị thêm frames giữa
        if len(frames) > 2:
            num_frames = min(4, len(frames))
            for i in range(num_frames):
                plt.subplot(2, num_frames//2, i+1)
                frame_idx = i * (len(frames) - 1) // (num_frames - 1)
                plt.imshow(frames[frame_idx])
                plt.title(f"Frame {frame_idx}\nPos: {positions[frame_idx]}\nAction: {actions[frame_idx] if frame_idx > 0 else 'None'}")
        else:
            plt.subplot(1, 2, 1)
            plt.imshow(frames[0])
            plt.title(f"First Frame\nPos: {positions[0]}")
            
            plt.subplot(1, 2, 2)
            plt.imshow(frames[-1])
            plt.title(f"Last Frame\nPos: {positions[-1]}")
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Đã lưu debug frames tại: {filename}")
        
        # Kiểm tra sự khác biệt giữa các frames
        different_pixels = []
        for i in range(1, len(frames)):
            diff = np.sum(frames[i] != frames[i-1])
            different_pixels.append(diff)
        
        print(f"Sự khác biệt giữa các frames liên tiếp (pixel): {different_pixels}")
    else:
        print("Không đủ frames để kiểm tra")
    
    return {
        "frames": frames,
        "rewards": rewards,
        "positions": positions,
        "states": states,
        "actions": actions[:len(rewards)]
    }

def plot_comparison_chart(data_dict, filename, title="Comparison", x_label="Algorithm", y_label="Value"):
    """
    Vẽ và lưu biểu đồ so sánh giữa các thuật toán.
    
    Args:
        data_dict (dict): Dictionary với khóa là tên thuật toán và giá trị là dữ liệu
        filename (str): Đường dẫn file lưu hình ảnh
        title (str): Tiêu đề biểu đồ
        x_label (str): Nhãn trục x
        y_label (str): Nhãn trục y
    """
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Tạo figure
    plt.figure(figsize=(10, 6))
    
    # Lấy tên thuật toán và giá trị
    algorithms = list(data_dict.keys())
    values = list(data_dict.values())
    
    # Vẽ biểu đồ cột
    bars = plt.bar(algorithms, values)
    
    # Thêm giá trị lên đỉnh cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05 * max(values),
                f'{height:.2f}', ha='center', va='bottom')
    
    # Thêm tiêu đề và nhãn
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Lưu hình ảnh
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def create_heat_map(grid, filename, title="Heatmap"):
    """
    Tạo và lưu heat map.
    
    Args:
        grid (array): Mảng 2D chứa dữ liệu heatmap
        filename (str): Đường dẫn file lưu hình ảnh
        title (str): Tiêu đề biểu đồ
    """
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Tạo figure
    plt.figure(figsize=(10, 8))
    
    # Vẽ heatmap
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    
    # Thêm tiêu đề
    plt.title(title)
    
    # Lưu hình ảnh
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def visualize_sensor_data(car_pos, sensor_data, directions, filename, grid_size=100, title="Sensor Data"):
    """
    Trực quan hóa dữ liệu cảm biến.
    
    Args:
        car_pos (array): Vị trí xe
        sensor_data (array): Dữ liệu cảm biến
        directions (list): Danh sách các hướng
        filename (str): Đường dẫn file lưu hình ảnh
        grid_size (int): Kích thước lưới
        title (str): Tiêu đề biểu đồ
    """
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Tạo figure
    plt.figure(figsize=(8, 8))
    
    # Vẽ biên
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    
    # Vẽ xe
    car = Circle((car_pos[0], car_pos[1]), 2, color='blue')
    plt.gca().add_patch(car)
    
    # Vẽ cảm biến
    for i, direction in enumerate(directions):
        dir_vector = np.array(direction) / np.linalg.norm(direction)
        end_point = car_pos + dir_vector * sensor_data[i]
        plt.plot([car_pos[0], end_point[0]], 
                [car_pos[1], end_point[1]], 'y-', alpha=0.7)
        
        # Thêm text giá trị
        mid_point = car_pos + dir_vector * sensor_data[i] * 0.5
        plt.text(mid_point[0], mid_point[1], f"{sensor_data[i]:.1f}", 
                fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Thêm tiêu đề
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Lưu hình ảnh
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def create_action_distribution_chart(action_counts, filename, title="Action Distribution"):
    """
    Tạo biểu đồ phân phối hành động.
    
    Args:
        action_counts (dict): Dictionary với khóa là tên hành động và giá trị là số lần thực hiện
        filename (str): Đường dẫn file lưu hình ảnh
        title (str): Tiêu đề biểu đồ
    """
    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Tạo figure
    plt.figure(figsize=(10, 6))
    
    # Lấy tên hành động và số lần
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    
    # Tính tổng số hành động
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]
    
    # Vẽ biểu đồ cột
    bars = plt.bar(actions, percentages)
    
    # Thêm giá trị lên đỉnh cột
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%\n({count})', ha='center', va='bottom')
    
    # Thêm tiêu đề và nhãn
    plt.title(title)
    plt.xlabel('Action')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, max(percentages) * 1.2)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Lưu hình ảnh
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()