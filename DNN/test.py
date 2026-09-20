import os
import sys

# 1. Cấu hình đường dẫn tuyệt đối tới thư mục network/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.join(BASE_DIR, 'network')

if network_dir not in sys.path:
    sys.path.append(network_dir)

import random
import torch
import matplotlib.pyplot as plt

# Import model_conf từ network_dir đã append ở trên
from network.model_conf import model_conf

# ---------------------------------------------------------
# 2. Định nghĩa đường dẫn file checkpoint & thiết bị
# ---------------------------------------------------------
model_path = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
data_path = os.path.join(BASE_DIR, "checkpoints", "fixed_test_data.pt")
output_img_path = os.path.join(BASE_DIR, "checkpoints", "test_comparison.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_and_visualize(num_samples=10):
    # Kiểm tra sự tồn tại của file checkpoint
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print("❌ Lỗi: Không tìm thấy file model hoặc test data tại thư mục checkpoints/!")
        print("Vui lòng chạy script training trước để tạo các file này.")
        return

    print("📂 Đang nạp mô hình và dữ liệu test...")
    
    # 1. Khởi tạo Model và nạp weights
    model = model_conf().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Nạp dữ liệu test cố định
    saved_data = torch.load(data_path, map_location='cpu')
    clean_all = saved_data['clean']
    noisy_all = saved_data['noisy']

    # 3. Chọn ngẫu nhiên 10 chỉ số từ 10,000 mẫu
    total_samples = clean_all.size(0)
    random_indices = random.sample(range(total_samples), num_samples)

    clean_samples = clean_all[random_indices].to(device)
    noisy_samples = noisy_all[random_indices].to(device)

    # 4. Cho ảnh qua model để khử nhiễu
    with torch.no_grad():
        logits, _, _ = model(noisy_samples, expected_classes=clean_samples)
        reconstructed_samples = torch.sigmoid(logits).cpu()

    clean_samples = clean_samples.cpu()
    noisy_samples = noisy_samples.cpu()

    # ---------------------------------------------------------
    # 5. Vẽ biểu đồ so sánh bằng Matplotlib
    # ---------------------------------------------------------
    print(f"🎨 Đang tiến hành hiển thị {num_samples} ảnh ngẫu nhiên...")
    
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

    for i in range(num_samples):
        # Hàng 1: Ảnh gốc ban đầu (Clean)
        axes[0, i].imshow(clean_samples[i].view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("1. Ảnh gốc (Clean)", fontsize=10, fontweight='bold', loc='left')

        # Hàng 2: Ảnh bị thêm nhiễu (Noisy)
        axes[1, i].imshow(noisy_samples[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("2. Ảnh nhiễu (Noisy)", fontsize=10, fontweight='bold', loc='left')

        # Hàng 3: Ảnh sau khi giảm nhiễu (Denoised)
        axes[2, i].imshow(reconstructed_samples[i].view(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title("3. Sau giảm nhiễu", fontsize=10, fontweight='bold', loc='left')

    plt.tight_layout()
    
    # Lưu kết quả ra file hình ảnh
    plt.savefig(output_img_path, bbox_inches='tight', dpi=300)
    print(f"🖼️ Đã lưu hình ảnh so sánh tại: {output_img_path}")
    
    # Hiển thị cửa sổ hình ảnh
    plt.show()


if __name__ == "__main__":
    test_and_visualize(num_samples=10)