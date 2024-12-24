import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# 加载.mat文件
mat_data = scipy.io.loadmat('Pavia.mat')
pavia = mat_data['pavia']
labels = scipy.io.loadmat('Pavia_gt.mat')['pavia_gt']
height, width, bands = pavia.shape
data_normalized = pavia.astype(np.float32) / np.max(pavia)
# 选择切片的大小
patch_size = 32

# 创建空列表，用来存储切片
image_patches = []
label_patches = []
# 切片操作：按行和列将图像和标签切分成 128x128 块
for i in range(0, height - patch_size + 1, patch_size):  # 高度方向切片
    for j in range(0, width - patch_size + 1, patch_size):  # 宽度方向切片
        # 获取每个图像块
        image_patch = pavia[i:i + patch_size, j:j + patch_size, :]
        # 获取每个标签块
        label_patch = labels[i:i + patch_size, j:j + patch_size]

        # 将切片转换为PyTorch张量
        image_patch_tensor = torch.tensor(image_patch, dtype=torch.float32)
        label_patch_tensor = torch.tensor(label_patch, dtype=torch.long)  # 标签通常是整数类型

        # 添加到patch列表中
        image_patches.append(image_patch_tensor)
        label_patches.append(label_patch_tensor)

# 将切片列表转换为张量形式
image_patches_tensor = torch.stack(image_patches)
label_patches_tensor = torch.stack(label_patches)

#几个Label
num_classes = 10
# 使用 one_hot 函数进行转换
one_hot_label_tensor = F.one_hot(label_patches_tensor, num_classes=num_classes)


print(f"图像切片后的数据形状: {image_patches_tensor.shape}")
print(f"标签切片后的数据形状: {label_patches_tensor.shape}")

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器部分 (下采样)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # 中间部分
        self.middle = self.conv_block(256, 512)

        # 解码器部分 (上采样)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # 最终输出层
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # 额外的卷积层调整通道数
        self.conv3 = self.conv_block(512, 256)  # 解码器中的通道数匹配
        self.conv2 = self.conv_block(256, 128)
        self.conv1 = self.conv_block(128, 64)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))

        # 中间部分
        middle = self.middle(F.max_pool2d(enc3, 2))

        # 解码器部分
        dec3 = self.decoder3(middle)
        dec3 = torch.cat([dec3, enc3], dim=1)  # 跳跃连接
        dec3 = self.conv3(dec3)  # 调整通道数

        dec2 = self.decoder2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)  # 跳跃连接
        dec2 = self.conv2(dec2)  # 调整通道数

        dec1 = self.decoder1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # 跳跃连接
        dec1 = self.conv1(dec1)  # 调整通道数

        # 输出层
        out = self.final(dec1)
        return out

# 实例化U-Net模型
model = UNet(in_channels=102, out_channels=10)  # 假设输入102通道的图像，输出10类
#print(model)

#切割数据
# 将 image_patches_tensor 和 one_hot_label_tensor 转换为 numpy 数组进行划分
image_patches_np = image_patches_tensor.numpy()
label_patches_np = one_hot_label_tensor.numpy()

# 使用 train_test_split 划分数据，test_size=0.2表示 20% 用于测试集
X_train, X_test, y_train, y_test = train_test_split(image_patches_np, label_patches_np, test_size=0.2, random_state=42)

# 将划分后的数据转换回 PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 自定义Dataset类
class HyperspectralDataset(Dataset):
    def __init__(self, images, labels):
        """
        初始化Dataset
        :param images: 输入图像数据 (Tensor)
        :param labels: 标签数据 (Tensor)
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        # 返回数据集的大小
        return len(self.images)

    def __getitem__(self, idx):
        # 返回单个样本和对应标签
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# 假设你已经将训练数据转换为 Tensor
# X_train_tensor 和 y_train_tensor 已经存在，确保它们是 PyTorch 张量
train_dataset = HyperspectralDataset(X_train_tensor, y_train_tensor)

# 创建 DataLoader，用于批量加载训练数据
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失


# 检查是否可以使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型和数据迁移到设备（GPU 或 CPU）
model = model.to(device)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 进入训练模式
    running_loss = 0.0

    for images, labels in train_loader:
        # 在输入模型之前，调整数据的维度顺序
        images = images.permute(0, 3, 1,
                                2)  # 将数据从 [batch_size, height, width, channels] 转换为 [batch_size, channels, height, width]
        labels = labels.permute(0, 3, 1, 2)  # 如果标签也是类似的情况，做相同处理
        labels = labels.long()
        # 将图像和标签迁移到 GPU（如果可用）
        images, labels = images.to(device), labels.to(device)



        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        # 转换标签为 Long 类型
        labels = labels.long()
        labels = labels.argmax(dim=1)
        #print(f"Labels shape: {labels.shape}, Labels dtype: {labels.dtype}")
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 累加损失
        running_loss += loss.item()

    # 打印每个 epoch 的损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

def evaluate_model(model, data_loader, device):
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 不计算梯度
        for images, labels in data_loader:
            # 数据预处理
            images = images.permute(0, 3, 1, 2).to(device)
            labels = labels.permute(0, 3, 1, 2).to(device)
            labels = labels.argmax(dim=1)

            # 前向传播
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 收集预测值和真实值
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 拼接所有批次的数据
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算准确率
    acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
    print(f"Test Accuracy: {acc:.4f}")

    # 打印分类报告
    print(classification_report(all_labels.flatten(), all_preds.flatten()))

    # 混淆矩阵
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels.flatten(), all_preds.flatten()))

# 创建测试集 DataLoader
test_dataset = HyperspectralDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 测试模型
evaluate_model(model, test_loader, device)

torch.save(model.state_dict(), 'unet_model.pth')
print("模型已保存！")

