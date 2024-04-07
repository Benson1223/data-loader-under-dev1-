# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import os

# class CustomDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data = pd.read_csv(csv_file, header=None)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
#         image = Image.open(img_name)
#         label = self.data.iloc[idx, 1]
#         if label == 'TRUE':
#             label = 1
#         else:
#             label = 0
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# # 資料集根目錄
# root_dir = './dataset/'
# # 轉換器，這裡只是一個範例，你可以根據你的需求增加更多的轉換
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # 建立訓練集
# train_dataset = CustomDataset(csv_file='datasets_label.csv',
#                               root_dir=root_dir,
#                               transform=transform)

# # 建立測試集
# test_dataset = CustomDataset(csv_file='datasets_label.csv',
#                              root_dir=root_dir,
#                              transform=transform)



# # 使用 DataLoader 加載資料
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, phase='train'):
        self.data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        self.data = self.data[self.data[2] == phase]  # Filter data based on phase (train/test)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        if label == 'TRUE':
            label = 1
        else:
            label = 0
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

# 資料集根目錄
root_dir = 'dataset'
# 轉換器，這裡只是一個範例，你可以根據你的需求增加更多的轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 建立訓練集
train_dataset = CustomDataset(csv_file='datasets_label.csv',
                              root_dir=root_dir,
                              transform=transform,
                              phase='Train')

# 建立測試集
test_dataset = CustomDataset(csv_file='datasets_label.csv',
                             root_dir=root_dir,
                             transform=transform,
                             phase='Test')

# 使用 DataLoader 加載資料
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 打印訓練資料路徑
print("Train data paths:")
for idx, (image, label, img_path) in enumerate(train_loader):
    for path in img_path:
        print(f"Index: {idx}, Path: {path}")

# 打印測試資料路徑
print("\nTest data paths:")
for idx, (image, label, img_path) in enumerate(test_loader):
    for path in img_path:
        print(f"Index: {idx}, Path: {path}")
