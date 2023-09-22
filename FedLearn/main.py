import copy
import parser
import numpy as np
import torch
import torch.multiprocessing as mp
from matplotlib import pyplot as plt

from torchvision import datasets, transforms

from Algorithm.FedAvg import FedAvg
from network import MLP
from option import args_parser
from sampling import mnist_iid, mnist_noniid
from test import test_img
from update import ClientUpdate


def main():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

def main_worker(args):
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

    elif args.dataset == 'fashion_mnist':
        trans_fashion = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('../data/mnist/', train=True, download=True, transform=trans_fashion)
        dataset_test = datasets.FashionMNIST('../data/mnist/', train=False, download=True, transform=trans_fashion)


    else:
        exit('Error: unrecognized dataset')
    image_size = dataset_train[0][0].shape

    if args.model == 'mlp':
        len_in = 1
        for x in image_size:
            len_in *= x
        net_global = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_global)
    net_global.train()
    w_global = net_global.state_dict()

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_global for i in range(args.num_users)]

    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_clients), 1)
        idxs_users = np.random.choice(range(args.num_clients), m, replace=False)
        for idx in idxs_users:
            local = ClientUpdate(args=args, dataset=dataset_train, index=dict_clients[idx])
            w, loss = local.train(net=copy.deepcopy(net_global).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        w_glob = FedAvg(w_locals)
        net_global.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        print('Communication round is ', iter + 1)
        loss_train.append(loss_avg)

    #plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_global.eval()
    acc_train, loss_train = test_img(net_global, dataset_train, args)
    acc_test, loss_test = test_img(net_global, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

if __name__ == '__main__':
    main()


