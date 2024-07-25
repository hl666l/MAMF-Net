import torch
import torch.nn.functional as F


def test(model, test_loader, criterion, test_device, epoch, y_true, y_prob_1):
    model.eval()
    with torch.no_grad():
        test_number = 0
        test_correct = 0.0
        test_total_loss = 0.0
        for step, (table_data, data, label) in enumerate(test_loader):
            y_true.extend(label.numpy())
            if torch.cuda.is_available():
                criterion = criterion.to(test_device)
                label = label.to(test_device)
                data = data.to(test_device)
                table_data = table_data.to(test_device)
            else:
                print("！！！！！！ GPU无法使用 ！！！！！！")

            y = model(table_data, data)
            test_loss = criterion(y, label)
            softmax_output = F.softmax(y.cpu(), dim=1)

            y_1 = softmax_output[:, 1].data.numpy()

            y_prob_1.extend(y_1)

            label = label.cpu()
            test_total_loss += test_loss.item()
            y = torch.max(y.cpu(), dim=1)[1].data.numpy()
            print("epoch:", epoch, y)

            test_correct += float((y == label.data.numpy()).astype(int).sum())
            test_number += label.size(0)
        test_accuracy = test_correct / float(test_number)
    return test_accuracy, test_total_loss, y_true, y_prob_1

