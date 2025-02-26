import unittest
import os
import sys

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Tìm tất cả các test case trong thư mục hiện tại
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Chạy các test case
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Kết thúc với mã lỗi thích hợp
    sys.exit(not result.wasSuccessful())