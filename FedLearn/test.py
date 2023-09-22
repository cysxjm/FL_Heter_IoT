import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from update import DatasetSplit


def test_img(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for index, (data,target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)

        test_loss += F.cross_entropy(log_probs,target,reduction='sum').item()

        y_pred = log_probs.data.max(1, keepdim = True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_lstm(net_g, dataset_test, args, test_indices):
    data_loader = DataLoader(DatasetSplit(dataset_test, test_indices), batch_size=40, shuffle=True, drop_last=True)
    losses_mse = []
    losses_mae = []
    for idx, (data, target) in enumerate(data_loader):
        data = data.detach().clone().type(torch.FloatTensor)
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        outputs = net_g(data)
        loss_MSE = torch.nn.MSELoss()
        loss_MAE = torch.nn.L1Loss()
        loss_mse = loss_MSE(outputs, torch.squeeze(target))
        loss_mae = loss_MAE(outputs, torch.squeeze(target))
        losses_mse.append(loss_mse.cpu().data.numpy())
        losses_mae.append(loss_mae.cpu().data.numpy())

    # losses_l1 = np.array(losses_mae)
    loss_mae_mean = np.mean(losses_mae)
    # losses_mse = np.array(losses_mse)
    loss_mse_mean = np.mean(losses_mse)
    # mean_l1 = np.mean(losses_l1) * max_speed
    # std_l1 = np.std(losses_l1) * max_speed

    # print('Tested: MSE_loss: {}, MAE_mean: {}'.format(loss_mse_mean, loss_mae_mean))
    return loss_mse_mean, loss_mae_mean

def test_module(net, dataset, idxs, args):
    if args.model == "lstm":
        return test_lstm(net, dataset, args, idxs)
    else:
        return test_img(net, dataset, args)