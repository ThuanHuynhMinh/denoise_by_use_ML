import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
network_dir = os.path.join(BASE_DIR, 'network')
sys.path.append(network_dir)

save_dir = os.path.join(BASE_DIR, "checkpoints")
save_path = os.path.join(save_dir, "best_model.pth")
data_dir = os.path.join(BASE_DIR, "data")  # Tự động lưu dữ liệu vào thư mục dự án

# Tạo trước các thư mục lưu trữ
os.makedirs(save_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets,transforms
import numpy as np
from network.model_conf import model_conf

def add_noise(images, noise_factor=0.3):

    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    
    # Kẹp giá trị trong khoảng [0.0, 1.0]
    return torch.clamp(noisy_images, 0.0, 1.0)

def train_epoch(model,optimizer,train_loader,device):
    model.train()
    total_loss = 0.0

    for datas,_ in train_loader:

        datas=datas.to(device)
        # Tạo ảnh bị nhiễu
        noisy_images = add_noise(datas, noise_factor=0.3).to(device)

        #reset gradient
        optimizer.zero_grad()

        logits, loss, _ = model(noisy_images, expected_classes=datas)

        #cap nhat trong so
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss=total_loss/len(train_loader)
    return epoch_loss

def test_epoch(model,test_loader,device):
    model.eval()
    total_loss=0

    with torch.no_grad():
        for noisy,clean in test_loader:
            noisy=noisy.to(device)
            clean=clean.to(device)


            logits, loss, _ = model(noisy, expected_classes=clean)
            


            total_loss += loss.item()

        epoch_loss=total_loss/len(test_loader)
        return epoch_loss
        


# ---------------------------------------------------------
# 1. Cấu hình Hyperparameters & Device
# ---------------------------------------------------------

EPOCHS=50
BATCH_SIZE=200
learning_rate=0.001
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. Chuẩn bị dữ liệu MNIST (Tự động download nếu chưa có)
# ---------------------------------------------------------

print("📥 Đang tải bộ dữ liệu MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(),                           # Chuyển ảnh PIL thành Tensor [1, 28, 28] thuộc [0.0, 1.0]
    transforms.Lambda(lambda x: x.view(-1))          # Duỗi phẳng từ [1, 28, 28] thành [784]
])
train_datasets=datasets.MNIST(root=data_dir,
                              train=True,
                              download=True,
                              transform=transform
                              )

test_datasets=datasets.MNIST(root= data_dir,
                              train=False,
                              download=True,
                              transform=transform
                              )

train_loader = DataLoader(
    dataset=train_datasets, 
    batch_size=BATCH_SIZE, 
    shuffle=True          # Xáo trộn dữ liệu mỗi epoch khi train   
)
fixed_clean = torch.stack([img for img, _ in test_datasets])
fixed_noisy = add_noise(fixed_clean, noise_factor=0.3)

# Lưu tập nhiễu cố định
noise_save_path = os.path.join(BASE_DIR, "checkpoints", "fixed_test_data.pt")

# Tạo chắc chắn thư mục checkpoints trước khi torch.save
os.makedirs(os.path.dirname(noise_save_path), exist_ok=True)
torch.save({
    'clean': fixed_clean,
    'noisy': fixed_noisy,
    'target': test_datasets.targets
}, noise_save_path)
print(f"💾 Đã lưu tập ảnh nhiễu cố định tại: {noise_save_path}")

fixed_test_dataset = TensorDataset(fixed_noisy, fixed_clean)
fixed_test_loader = DataLoader(fixed_test_dataset, batch_size=64, shuffle=False)
# ---------------------------------------------------------
# 3. Khởi tạo Model & Optimizer
# ---------------------------------------------------------

model=model_conf().to(device)

optimizer=optim.Adam(model.parameters(),
                     lr=learning_rate,
                     weight_decay=model.model_para.l2_lamda
                     )


# ---------------------------------------------------------
# 4. Vòng lặp Huấn luyện (Training Loop)
# ---------------------------------------------------------
print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")

best_loss=float('inf')
for epoch in range(1,EPOCHS+1):
    train_loss = train_epoch(model, optimizer,train_loader, device)
    # 2. Đánh giá trên tập test cố định
    test_loss = test_epoch(model, fixed_test_loader, device)

    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # 3. Lưu lại Checkpoint của Model khi Test Loss đạt giá trị nhỏ nhất
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), save_path)
        print(f"  --> 💾 Đã lưu best checkpoint với Test Loss: {best_loss:.4f}")

print(f"\n✅ Hoàn thành huấn luyện!")
print(f"🏆 Test Loss tốt nhất đạt được: {best_loss:.4f}")
print(f"💾 Model tốt nhất đã được lưu tại: {save_path}")



