import copy
import os
import random
import math
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torch

import time

from Algorithm.FedAvg import FedAvg
from LOOPDataset import LOOPDataset
from test import test_img, test_lstm
from update import ClientUpdate
from network import MLP, LSTM
from option import args_parser
from sampling import mnist_iid, mnist_noniid, iid_onepass


def sum_list(list):
    for i in range(len(list)):
        if i == 0:
            list[i] = list[i]
        else:
            list[i] = list[i - 1] + list[i]
    return list


def generate_heterogeneous(var, mean, num):
    target = var
    mean = mean
    std = np.sqrt(target)
    alpha = (mean ** 2 - mean ** 3) / std ** 2 - mean
    beta = alpha / mean - alpha
    sampler = torch.distributions.beta.Beta(alpha, beta)
    output = torch.tensor([sampler.sample()]).view(1, -1)
    eps = 0.0001
    for i in range(num - 1):
        var_old = output.var()
        output = torch.cat([output, sampler.sample().view(1, -1)], dim=0)
        var_new = output.var()
        while np.abs(var_new - target) > eps and np.abs(var_new - target) > np.abs(var_old - target) + target / 500:
            output = output[:-1]
            output = torch.cat([output, sampler.sample().view(1, -1)], dim=0)
            var_new = output.var()
    return output.squeeze().tolist()


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Load dataset and sample it for each client
    if args.dataset == 'mnist':
        train_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=train_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=train_mnist)
        if args.iid:

            dict_clients = mnist_iid(dataset_train, args.num_clients)
            print('Now the dataset is i.i.d')
        else:
            dict_clients = mnist_noniid(dataset_train, args.num_clients)
            print('Now the dataset is non_i.i.d')

    elif args.dataset == 'loop' and args.iid:
        dataset_train = LOOPDataset(data_path="../data/loop/", phase='train')
        dataset_test = LOOPDataset(data_path="../data/loop/", phase='eval')
        dataset_test_size = int(args.dataset_train_size * 0.25)
        dict_clients, dict_test = iid_onepass(dataset_train, args.dataset_train_size, dataset_test, dataset_test_size,
                                             args.num_clients, dataset_name="loop")

    else:
        exit('Error: unrecognized dataset')
    image_size = dataset_train[0][0].shape

    # Build network model

    if args.model == 'mlp':
        len_in = 1
        for x in image_size:
            len_in *= x
        net_global = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

    elif args.model == 'lstm' and args.dataset == 'loop':
        net_global  = LSTM(image_size[1], image_size[1], image_size[1], output_last=True)

    else:
        exit('Error: unrecognized model')
    print(net_global)
    net_global.train()
    w_global = net_global.state_dict()

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_global for i in range(args.num_users)]

    loss_train = []
    acc_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    epoch_time = []
    comm = []
    test_users = []
    if args.heter == 0:
        heter = [args.heter_avg]*args.num_clients
    elif args.heter == 0.25:
        if (args.num_clients/2) % 2 == 0:
            a = [1]*((args.num_clients)/2)
            b = [0]*((args.num_clients)/2)
            heter = a + b
        else:
            a = [1] * math.floor((args.num_clients) / 2)
            c = [0.5]
            b = [0] * math.floor((args.num_clients) / 2)
            heter = a + b + c

    else:
        heter = generate_heterogeneous(args.heter, args.heter_avg, args.num_clients)

    # df = pd.DataFrame(columns=['step', 'loss_avg', 'acc_avg'])
    # df.to_csv('./fed_{}_{}_{}_C{}_iid{}_heter{}_H{}.csv'.format(args.dataset, args.model, args.epochs, args.frac,
    #                                                                   args.iid, args.heterogeneity, args.heter), mode='a',index=False)

    for iter in range(args.epochs):

        time_all = 0
        loss_locals = []
        acc_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_clients), 1)
        idxs_users = np.random.choice(range(args.num_clients), m, replace=False)
        T1 = time.perf_counter()
        #pool = multiprocessing.Pool(processes=m)
        for idx in idxs_users:
            #local =pool.apply_async(ClientUpdate(args=args, dataset=dataset_train, index=dict_clients[idx]))
            #w, loss, acc = pool.apply_async(local.train(net=copy.deepcopy(net_global).to(args.device)))
            local = ClientUpdate(args=args, dataset=dataset_train, index=dict_clients[idx])
            w, loss= local.local_update(net=copy.deepcopy(net_global).to(args.device))
            #w, loss= local.train(net=copy.deepcopy(net_global).to(args.device))
            #local = ClientUpdate(args=args, dataset=dataset_train, index=dict_clients[idx])
            #w, loss, acc = local.train(net=copy.deepcopy(net_global).to(args.device))
            # if args.all_clients:
            #   w_locals[idx] = copy.deepcopy(w)
            # else:
            if args.heterogeneity:
                plr = random.random()
                if plr > heter[idx]:
                    w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))
                    #acc_locals.append(copy.deepcopy(acc))
                    #print('{:3d} packet loss not happened the plr {:.3f} larger than pl {:.3f}'.format(idx ,plr, heter[idx]))
                else:
                    w_locals
                    print('{:3d} packet loss happened the plr {:.3f} less than pl {:.3f}'.format(idx ,plr, heter[idx]))
                #time.sleep(heter[idx]*1000)
                #time.sleep(heter[idx]*1000)
            else:
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                #cc_locals.append(copy.deepcopy(acc))
                # print('w_locals is', w_locals)
        T2 = time.perf_counter()
        print('The length of w is ', len(w_locals))
        if len(w_locals) != 0:
            w_glob = FedAvg(w_locals)
            loss_avg = sum(loss_locals) / len(loss_locals)
            #acc_avg = sum(acc_locals) / len(acc_locals)
        epoch_time.append((T2 - T1) / 60)
        net_global.load_state_dict(w_glob)
        comm.append(iter + 1)
        print('Round {:3d}, Average loss {:.3f} '.format(comm[iter], loss_avg))#Average Test accuracy {:.3f}acc_avg
        print('Communication round is ', comm[iter])
        print('The time consume is ', (T2 - T1) / 60)
        # print('Time consume for this round is', )
        # listw = [iter + 1, loss_avg, acc_avg]
        # df = pd.DataFrame([listw],columns=['step', 'loss_avg', 'acc_avg'])
        # print('Df is ', df)
        # df.to_csv('./fed_{}_{}_{}_C{}_iid{}_heter{}_H{}.csv'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.heterogeneity, args.heter)
        # ,mode='a',index=False)
        loss_train.append(loss_avg)
        #acc_train.append(acc_avg)

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(r'D:\Study\result\train_loss_round_fed_{}_{}_{}_C{}_iid{}_heter{}_H{}_Avg{}.png'.format(args.dataset, args.model,
                                                                                         args.epochs,
                                                                                         args.frac, args.iid,
                                                                                         args.heterogeneity,
                                                                                         args.heter ,args.heter_avg))
    # plt.savefig('./save/train_loss_round_fed_{}_{}_{}_C{}_iid{}_heter{}_H{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.heterogeneity, args.heter))

    # plt.figure()
    # plt.plot(range(len(acc_train)), acc_train)
    # plt.ylabel('acc_train')
    # plt.savefig(
    #     r'D:\Study\result\acc_train_round_fed_{}_{}_{}_C{}_iid{}_heter{}_H{}_Avg{}.png'.format(args.dataset, args.model, args.epochs,
    #                                                                             args.frac, args.iid, args.heterogeneity,
    #                                                                             args.heter ,args.heter_avg))
    # plt.savefig('./save/acc_train_round_fed_{}_{}_{}_C{}_iid{}_heter{}_H{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.heterogeneity, args.heter))

    epoch_time_sum = sum_list(epoch_time)

    plt.figure()
    plt.plot(epoch_time_sum, loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('time(min)')
    plt.savefig(
        r'D:\Study\result\train_loss_time_fed_{}_{}_{}_C{}_iid{}_heter{}_H{}_Avg{}.png'.format(args.dataset, args.model, args.epochs,
                                                                                args.frac, args.iid, args.heterogeneity,
                                                                                args.heter ,args.heter_avg))

    # plt.figure()
    # plt.plot(epoch_time_sum, acc_train)
    # plt.ylabel('acc_train')
    # plt.xlabel('time(min)')
    # plt.savefig(
    #     r'D:\Study\result\acc_train_time_fed_{}_{}_{}_C{}_iid{}_heter{}_H{}_Avg{}.png'.format(args.dataset, args.model, args.epochs,
    #                                                                            args.frac, args.iid, args.heterogeneity,
    #                                                                            args.heter ,args.heter_avg))

    # testing
    net_global.eval()
    if args.dataset == 'loop':
        test_indices = test_users[idx]
        test_lstm(net_global, dataset_test, args, test_indices)
    else:
        acc_train_eval, loss_train_eval = test_img(net_global, dataset_train, args)
        acc_test_eval, loss_test_eval = test_img(net_global, dataset_test, args)
    print("Training accuracy: {:.2f}, Training Loss: {:.2f}".format(acc_train_eval, loss_train_eval))
    print("Testing accuracy: {:.2f}, Testing Loss: {:.2f}".format(acc_test_eval, loss_test_eval))

    column_names_training = ['communication', 'time', 'loss_train', 'acc_train']
    df_training = pd.DataFrame(list(zip(comm, epoch_time_sum, loss_train, acc_train)), columns=column_names_training)
    evaluation = [acc_train_eval, loss_train_eval, acc_test_eval, loss_test_eval]
    column_names_eval = ['acc_train_eval', 'loss_train_eval', 'acc_test_eval', 'loss_test_eval']
    df_eval = pd.DataFrame([evaluation], columns=column_names_eval)
    df_training.to_csv(
        r'D:\Study\result\trafed_{}_{}_{}_C{}_iid{}_heter{}_H{}_Avg{}.csv'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                    args.iid, args.heterogeneity, args.heter ,args.heter_avg),
        index=False)
    df_eval.to_csv(
        r'D:\Study\result\evalfed_{}_{}_{}_C{}_iid{}_heter{}_H{}_Avg{}.csv'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                     args.iid, args.heterogeneity, args.heter ,args.heter_avg),
        index=False)
