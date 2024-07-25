import sys

sys.path.append("../DataProcess")
sys.path.append("../Model")
sys.path.append("../Tools")
import torch
import wandb
import argparse
from Model.MAMF import MAMF as Model
import Train
import Test
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from Creat_Dataset import create_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn.init as init

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, help="批次大小")
parser.add_argument('--image_lr', type=float, help="学习率")
parser.add_argument('--image_eta_min', type=float, help="学习率")
parser.add_argument('--image_T_max', type=int, help="批次大小")
parser.add_argument('--project_name', type=str, help="项目名称")
parser.add_argument('--device', type=str, help="运行设备")
parser.add_argument('--run_name', type=str, help="此次运行名称")

# 解析参数--------------------------------------------------------------------------
args = parser.parse_args()
batch_size = args.batch_size
# 原始学习率
lr = args.image_lr
# 学习率最小降到多少
eta_min = args.image_eta_min
T_max = args.image_T_max
project_name = args.project_name
device = args.device
run_name = args.run_name
num_epoch = 200
# 定量配置部分-----------------------------------------------------------------
input_channels = 10
species = 2
train_files_folder = "xxx"
test_files_folder = "xxx"
table_csv = "xxx"
# 数据预处理操作---------------------------------------------------------------
mean = (0.5,) * input_channels
std = (0.5,) * input_channels
data_transform = transforms.Compose([
    transforms.ToTensor(),  # 将数据转换为张量
])

# 创建train数据集-------------------------------------------
train_custom_dataset = create_dataset(train_files_folder, table_csv, data_transform)
train_loader = DataLoader(train_custom_dataset, batch_size, shuffle=True, num_workers=8)
# 创建test数据集---------------------------------------------
test_custom_dataset = create_dataset(test_files_folder, table_csv, data_transform)
test_loader = DataLoader(test_custom_dataset, batch_size, shuffle=False)
# -----------------------------------
config = {
    "image_dataset": train_files_folder,
    "table_data": table_csv,
    "batch_size": batch_size
}
wandb.init(
    project=project_name,
    name=run_name,
    config=config
)
model = Model()
"""---------------------------------参数初始化-----------------------------"""
layers_to_exclude = ['layer1', 'layer2', 'layer3', 'layer4']
for name, param in model.named_parameters():
    if all(exclude_layer not in name for exclude_layer in layers_to_exclude):
        if 'weight' in name:
            init.normal_(param.data, mean=0, std=0.01)
        elif 'bias' in name:
            init.constant_(param.data, val=0.0)
"""---------------------------------参数加载-------------------------------"""
model2 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
pre_weights = model2.state_dict()
model_dict = model.state_dict()
for name, param in pre_weights.items():
    if (name in model_dict) and name != "fc.weight" and name != "fc.bias" and name != "conv1.weight":
        model_dict[name].copy_(param)
"""-----------------------------------------------------------------------"""
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=lr)
"""---------函数配置----------------------------"""
criterion = torch.nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max, eta_min)
"""--------------训练--------------------------"""
if __name__ == '__main__':
    for epoch in range(num_epoch):
        y_true = []
        y_probs_1 = []
        model = Train.train(model, train_loader, criterion, optimizer, device)
        test_accuracy, test_loss, y_true, y_probs_1 = Test.test(model, test_loader, criterion, device,
                                                                epoch, y_true, y_probs_1)
        scheduler.step()
        roc_auc = roc_auc_score(y_true, y_probs_1)
        wandb.log({"test_accuracy": test_accuracy, "roc_auc": roc_auc})
    wandb.finish()
