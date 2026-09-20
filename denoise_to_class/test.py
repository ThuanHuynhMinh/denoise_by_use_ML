import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

save_dir=os.path.join(BASE_DIR,"checkpoints")
denoise_path=os.path.join(save_dir,"model_denoise.pth")
class_path=os.path.join(save_dir,"model_class.pth")
data_path= os.path.join(save_dir,"fixed_test_data.pt")

import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from denoise.model_conf_denoise import model_conf as DenoiseModel
from classify.model_conf_class import model_conf as ClassiModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(denoise_path) or not os.path.exists(data_path) or not os.path.exists(class_path):
    print("❌ Lỗi: Không tìm thấy file model hoặc test data tại thư mục checkpoints/!")
    print("Vui lòng chạy script training trước để tạo các file này.")
    sys.exit()
    


#Khoi tao model va nap parameters
model_denoise=DenoiseModel().to(device)
model_denoise.load_state_dict(torch.load(denoise_path, map_location=device,weights_only=True))
model_denoise.eval()
model_classi=ClassiModel().to(device)
model_classi.load_state_dict(torch.load(class_path, map_location=device,weights_only=True))
model_classi.eval()

# 2. Nạp dữ liệu test cố định
saved_data = torch.load(data_path, map_location='cpu',weights_only=True)
clean_all = saved_data['clean']
noisy_all = saved_data['noisy']
target_all = saved_data['target']

# 3. Đóng gói dữ liệu vào TensorDataset & DataLoader
test_dataset = TensorDataset(noisy_all, clean_all, target_all)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Biến tích lũy số lượng dự đoán đúng
correct_noisy = 0      # Đúng khi dự đoán trực tiếp trên ảnh nhiễu
correct_denoised = 0   # Đúng khi dự đoán trên ảnh đã qua Denoising Autoencoder
total_samples = 0

print("\n🚀 Đang tiến hành đánh giá hiệu năng Pipeline...")

with torch.no_grad(): # Tắt gradient để tối ưu bộ nhớ & tốc độ
    for noisy_imgs, clean_imgs, labels in test_loader:
        # Chuyển batch dữ liệu sang thiết bị (GPU/CPU)
        noisy_imgs = noisy_imgs.to(device)
        labels = labels.to(device)
        
        denoised_logits = model_denoise(noisy_imgs)
        denoised_imgs = torch.sigmoid(denoised_logits)
        
        # B1. Phân loại trực tiếp trên ảnh nhiễu (Chưa lọc)
        outputs_noisy = model_classi(noisy_imgs)
        preds_noisy = outputs_noisy.argmax(dim=1)
        correct_noisy += (preds_noisy == labels).sum().item()
        
        # B2. Phân loại trên ảnh đã qua lọc nhiễu (Denoised)
        outputs_denoised = model_classi(denoised_imgs)
        preds_denoised = outputs_denoised.argmax(dim=1)
        correct_denoised += (preds_denoised == labels).sum().item()
        
        total_samples += labels.size(0)

# 5. Tính toán và in ra Accuracy
acc_noisy = (correct_noisy / total_samples) * 100
acc_denoised = (correct_denoised / total_samples) * 100

print("=" * 50)
print(f"📊 KẾT QUẢ ĐÁNH GIÁ TRÊN {total_samples} MẪU TEST:")
print(f" ❌ Ảnh bị nhiễu (Noisy Input)        : {acc_noisy:.2f}%")
print(f" ✅ Ảnh đã khử nhiễu (Denoised Input): {acc_denoised:.2f}%")
print(f" 📈 Mức độ cải thiện                  : +{acc_denoised - acc_noisy:.2f}%")
print("=" * 50)

import matplotlib.pyplot as plt

# Lấy thử 1 sample từ test_loader
sample_noisy, sample_clean, _ = next(iter(test_loader))
sample_noisy = sample_noisy.to(device)

with torch.no_grad():
    sample_denoised = model_denoise(sample_noisy)

# Vẽ 3 ảnh so sánh
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(sample_clean[0].cpu().view(28, 28), cmap='gray')
axes[0].set_title("Clean (Gốc)")

axes[1].imshow(sample_noisy[0].cpu().view(28, 28), cmap='gray')
axes[1].set_title("Noisy (Nhiễu)")

axes[2].imshow(sample_denoised[0].cpu().view(28, 28), cmap='gray')
axes[2].set_title("Denoised (Lọc)")

plt.show()

print("Range Clean   :", clean_all.min().item(), "đến", clean_all.max().item())
print("Range Noisy   :", noisy_all.min().item(), "đến", noisy_all.max().item())
print("Range Denoised:", denoised_imgs.min().item(), "đến", denoised_imgs.max().item())