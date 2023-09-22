import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class ClientUpdate(object):
    def __init__(self, args, dataset=None, index=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset,index), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        epoch_acc = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_acc = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                preds = torch.argmax(log_probs, dim=1)
                acc = torch.sum(preds == labels).item()/len(preds)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_acc.append(acc)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            print(epoch_loss)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

    #def train_lstm(net, dataset, idxs, local_ep, device):
    def train_lstm(self, net):
        #optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-5)
        optimizer = torch.optim.RMSprop(net.parameters(), lr=self.args.lr)
        #ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=40, shuffle=True, drop_last=True)
        loss_func = nn.MSELoss()
        losses_train = []

        losses_epochs_train = []
        for epoch in range(self.args.local_ep):
            losses_epoch_train = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data = data.detach().clone().type(torch.FloatTensor)
                data, labels = data.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                outputs = net(data)
                loss_train = loss_func(outputs, torch.squeeze(labels))
                losses_train.append(loss_train.data)

                losses_epoch_train.append(loss_train.data)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
            # print(avg_losses_epoch_train)
            losses_epochs_train.append(avg_losses_epoch_train.tolist())
        # print(losses_epochs_train)
        return net.state_dict(), losses_epochs_train
    def local_update(self, net):
        if self.args.model == "lstm":
            return ClientUpdate.train_lstm(self, net)
        else:
            return ClientUpdate.train(self, net)
