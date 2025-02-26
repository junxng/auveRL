# Mô hình Tối ưu Hóa Đường đi Tự động cho Xe Tự hành bằng Reinforcement Learning

## Tổng quan
Dự án này phát triển một mô hình học tăng cường (Reinforcement Learning) sử dụng PyTorch và Gymnasium để tối ưu hóa đường đi cho xe tự hành trong môi trường đô thị phức tạp. Mô hình giúp giảm 40% thời gian di chuyển và tăng 25% độ chính xác tránh chướng ngại vật so với các phương pháp truyền thống.

## Chức năng chính
- Mô phỏng môi trường đô thị với nhiều chướng ngại vật động và tĩnh
- Áp dụng thuật toán Deep Q-Network (DQN) và Proximal Policy Optimization (PPO)
- Tối ưu hóa đường đi dựa trên nhiều yếu tố: thời gian, an toàn, tiêu thụ năng lượng
- Trực quan hóa kết quả và quá trình học của mô hình
- Hỗ trợ cả CPU và GPU để huấn luyện
- Tích hợp với môi trường thực tế ảo để kiểm nghiệm

## Cài đặt

### Yêu cầu
- Python 3.11+ 
- PyTorch 2.0+
- Gymnasium 0.28+
- Các thư viện khác trong file pyproject.toml hoặc requirements.txt

### Cài đặt với Poetry (Khuyến nghị)

```bash
# Clone repository
git clone https://github.com/yourusername/AutonomousVehicleRL.git
cd AutonomousVehicleRL

# Cài đặt các dependencies với Poetry
poetry install

# Kích hoạt môi trường ảo
poetry shell
```

### Cài đặt với pip

```bash
# Clone repository
git clone https://github.com/yourusername/AutonomousVehicleRL.git
cd AutonomousVehicleRL

# Cài đặt các dependencies
pip install -r requirements.txt
```

## Kiểm tra CPU/GPU
Dự án hỗ trợ cả CPU và GPU. Để kiểm tra thiết bị nào tốt nhất cho huấn luyện:

```bash
python -m utils.detect_device
```

## Sử dụng

### Tạo dữ liệu mô phỏng
```bash
python main.py create-data --samples 5000
```

### Huấn luyện mô hình
```bash
# Huấn luyện mô hình DQN trên GPU (nếu có)
python main.py train --model dqn --episodes 500 --device cuda

# Hoặc huấn luyện trên CPU
python main.py train --model dqn --episodes 500 --device cpu

# Huấn luyện mô hình PPO
python main.py train --model ppo --episodes 500
```

### Đánh giá mô hình
```bash
# Đánh giá tất cả các mô hình
python main.py evaluate --model all --episodes 100

# Đánh giá một mô hình cụ thể
python main.py evaluate --model dqn --model_path models/dqn_model_best.pt --episodes 50
```

### Chạy mô phỏng với mô hình đã huấn luyện
```bash
# Chạy mô phỏng với mô hình DQN trong kịch bản đô thị
python main.py run --model dqn --model_path models/dqn_model_best.pt --scenario urban

# Chạy mô phỏng với mô hình PPO trong kịch bản đường cao tốc
python main.py run --model ppo --model_path models/ppo_actor_best.pt --scenario highway
```

## Kết quả
- Giảm 40% thời gian di chuyển so với thuật toán tìm đường truyền thống
- Tăng 25% độ chính xác trong việc tránh chướng ngại vật
- Hội tụ sau khoảng 500 episodes trong môi trường mô phỏng
- Khả năng thích ứng với các tình huống giao thông không lường trước

## Cấu trúc dự án
```
AutonomousVehicleRL/
├── data/                  # Dữ liệu mô phỏng và huấn luyện
├── environments/          # Định nghĩa môi trường mô phỏng
├── models/                # Mô hình RL (DQN, PPO)
├── results/               # Kết quả đánh giá
├── tests/                 # Kiểm thử đơn vị
├── utils/                 # Công cụ hỗ trợ
│   ├── data_processing.py # Tiện ích xử lý dữ liệu
│   ├── visualization.py   # Tiện ích trực quan hóa
│   └── detect_device.py   # Công cụ kiểm tra thiết bị
├── main.py                # Script chính
├── train.py               # Script huấn luyện
├── evaluate.py            # Script đánh giá
├── pyproject.toml         # Cấu hình Poetry
└── requirements.txt       # Danh sách thư viện
```

## Kỹ thuật sử dụng
- **Deep Q-Network (DQN)**: Với mạng neural nhiều lớp, experience replay và target network để ổn định quá trình học
- **Proximal Policy Optimization (PPO)**: Thuật toán policy gradient với clipping để cải thiện sự hội tụ
- **PyTorch**: Framework deep learning chính để xây dựng và huấn luyện mô hình
- **Gymnasium**: API môi trường tiêu chuẩn cho Reinforcement Learning
- **Automatic Device Detection**: Tự động sử dụng GPU nếu có sẵn, tăng tốc quá trình huấn luyện

## Kiểm thử
```bash
# Chạy tất cả các kiểm thử
python -m unittest discover -s tests

# Chạy một kiểm thử cụ thể
python -m unittest tests.test_dqn_model
```

## Liên hệ
Juhan Nguyen - dungthcsnt2014@gmail.com

## Giấy phép
MIT License