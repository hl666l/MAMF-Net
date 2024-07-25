import torch


def train(model, train_loader, criterion, optimizer, train_device):
    # 训练
    model.train()
    for step, (data, data2, label) in enumerate(train_loader):
        #  GPU训练
        if torch.cuda.is_available():
            model = model.to(train_device)
            criterion = criterion.to(train_device)
            label = label.type(torch.LongTensor)
            data = data.to(train_device)
            data2 = data2.to(train_device)
            label = label.to(train_device)
        else:
            print("！！！！！！ GPU无法使用 ！！！！！！")
        y = model(data, data2)
        loss = criterion(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

