import torch
import numpy as np
import time
import platform
import psutil
import os

def get_system_info():
    """Thu thập thông tin hệ thống cơ bản."""
    info = {
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "CPU": platform.processor(),
        "RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "PyTorch": torch.__version__
    }
    return info

def get_gpu_info():
    """Thu thập thông tin GPU nếu có."""
    if not torch.cuda.is_available():
        return {"Available": False}
    
    info = {
        "Available": True,
        "Device Count": torch.cuda.device_count(),
        "Current Device": torch.cuda.current_device(),
        "Device Name": torch.cuda.get_device_name(0),
    }
    
    # Thử lấy thông tin bộ nhớ GPU nếu có thể
    try:
        info["Memory Allocated"] = f"{torch.cuda.memory_allocated(0) / (1024**3):.2f} GB"
        info["Memory Reserved"] = f"{torch.cuda.memory_reserved(0) / (1024**3):.2f} GB"
    except:
        info["Memory Info"] = "Not available"
    
    return info

def run_performance_test(device_name="auto", matrix_size=5000, iterations=5):
    """
    Chạy kiểm tra hiệu năng đơn giản trên CPU và GPU (nếu có).
    
    Args:
        device_name: 'cpu', 'cuda', hoặc 'auto' để tự phát hiện
        matrix_size: kích thước ma trận vuông để nhân
        iterations: số lần lặp lại để tính trung bình
    
    Returns:
        dict: Kết quả kiểm tra hiệu năng
    """
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    print(f"Running performance test on {device}...")
    results = {"device": str(device)}
    
    # Ma trận nhỏ hơn cho thử nghiệm nhanh
    small_size = min(matrix_size, 1000)
    
    # Thử nghiệm với ma trận nhỏ
    timings = []
    for _ in range(iterations):
        a = torch.randn(small_size, small_size, device=device)
        b = torch.randn(small_size, small_size, device=device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        c = torch.matmul(a, b)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        timings.append(time.time() - start_time)
    
    results["small_matrix_time"] = f"{np.mean(timings):.6f} seconds"
    
    # Thử nghiệm với ma trận lớn (nếu không phải thử nghiệm nhanh)
    if matrix_size > small_size:
        try:
            timings = []
            for _ in range(max(1, iterations // 2)):  # Ít lần lặp hơn cho ma trận lớn
                a = torch.randn(matrix_size, matrix_size, device=device)
                b = torch.randn(matrix_size, matrix_size, device=device)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.time()
                c = torch.matmul(a, b)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                timings.append(time.time() - start_time)
            
            results["large_matrix_time"] = f"{np.mean(timings):.6f} seconds"
        except RuntimeError as e:
            # Bắt lỗi nếu ma trận quá lớn (thường là out of memory)
            results["large_matrix_error"] = str(e)
    
    return results

def suggest_device():
    """Đề xuất device tốt nhất cho huấn luyện."""
    if torch.cuda.is_available():
        # Chạy một kiểm tra nhỏ để xác nhận GPU hoạt động tốt
        try:
            cpu_results = run_performance_test("cpu", matrix_size=1000, iterations=2)
            gpu_results = run_performance_test("cuda", matrix_size=1000, iterations=2)
            
            cpu_time = float(cpu_results["small_matrix_time"].split()[0])
            gpu_time = float(gpu_results["small_matrix_time"].split()[0])
            
            if gpu_time < cpu_time:
                return "cuda", f"GPU nhanh hơn CPU {cpu_time/gpu_time:.1f}x"
            else:
                return "cuda", "GPU khả dụng nhưng có thể không nhanh hơn CPU đáng kể"
        except Exception as e:
            return "cpu", f"GPU gặp vấn đề: {str(e)}"
    
    return "cpu", "Không tìm thấy GPU khả dụng"

def save_report(filename="device_report.txt"):
    """Lưu báo cáo đầy đủ về thiết bị."""
    system_info = get_system_info()
    gpu_info = get_gpu_info()
    cpu_results = run_performance_test("cpu", matrix_size=1000, iterations=3)
    
    recommended_device, reason = suggest_device()
    
    with open(filename, "w") as f:
        f.write("=== BÁO CÁO THIẾT BỊ CHO DỊCH VỤ HUẤN LUYỆN ===\n\n")
        
        f.write("Thông tin hệ thống:\n")
        for key, value in system_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Thông tin GPU:\n")
        for key, value in gpu_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Kết quả kiểm tra CPU:\n")
        for key, value in cpu_results.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        if gpu_info["Available"]:
            gpu_results = run_performance_test("cuda", matrix_size=1000, iterations=3)
            f.write("Kết quả kiểm tra GPU:\n")
            for key, value in gpu_results.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        f.write(f"Thiết bị đề xuất: {recommended_device}\n")
        f.write(f"Lý do: {reason}\n")
    
    print(f"Báo cáo thiết bị đã được lưu vào {filename}")
    return filename

if __name__ == "__main__":
    print("\n=== THÔNG TIN THIẾT BỊ CHO HUẤN LUYỆN ===\n")
    
    print("Thông tin hệ thống:")
    for key, value in get_system_info().items():
        print(f"  {key}: {value}")
    print()
    
    print("Thông tin GPU:")
    gpu_info = get_gpu_info()
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    print()
    
    device, reason = suggest_device()
    print(f"Thiết bị được đề xuất: {device}")
    print(f"Lý do: {reason}")
    print()
    
    save_choice = input("Bạn có muốn tạo báo cáo chi tiết? (y/n): ")
    if save_choice.lower() == 'y':
        report_file = save_report()
        print(f"Đã lưu báo cáo chi tiết vào {report_file}")
    
    use_choice = input(f"Bạn có muốn sử dụng '{device}' cho huấn luyện? (y/n): ")
    if use_choice.lower() == 'y':
        print(f"\nĐể sử dụng {device}, thêm tham số --device {device} khi chạy lệnh train hoặc evaluate.")
        print("Ví dụ: python main.py train --model dqn --episodes 500 --device", device)
    else:
        alt_device = "cpu" if device == "cuda" else "cuda"
        print(f"\nĐể sử dụng {alt_device}, thêm tham số --device {alt_device} khi chạy lệnh train hoặc evaluate.")
        print("Ví dụ: python main.py train --model dqn --episodes 500 --device", alt_device)