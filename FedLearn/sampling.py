import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_client):

    num_item = int(len(dataset)/num_client)
    dict_clients = {}
    all_index = [i for i in range(len(dataset))]

    for i in range (num_client):
        dict_clients[i] = set(np.random.choice(all_index,num_item,replace=False))
        all_index = list(set(all_index) - dict_clients[i])
    return dict_clients

def mnist_noniid(dataset, num_client):

    num_shard, num_imgs = 200, 300
    index_shard = [i for i in range(num_shard)]
    dict_clients = {i: np.array([], dtype='int64') for i in range(num_client)}
    index = np.arange(num_shard*num_imgs)
    labels = dataset.train_labels.numpy()

    index_labels = np.vstack((index, labels))
    index_labels = index_labels[:,index_labels[1,:].argsort()]
    index = index_labels[0,:]

    for i in range (num_client):
        rand_set = set(np.random.choice(index_shard, 2, replace=False))
        index_shard = list(set(index_shard)-rand_set)
        for rand in rand_set:
            dict_clients[i] = np.concatenate((dict_clients[i], index[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_clients

def get_indices(labels, user_labels, n_samples):
    indices = []
    for selected_label in user_labels:
        label_samples = np.where(labels[1, :] == selected_label)
        label_indices = labels[0, label_samples]
        selected_indices = list(np.random.choice(label_indices[0], n_samples, replace=False))
        indices += selected_indices
    return indices


def get_samples(indices, n_samples):
    selected_indices = list(np.random.choice(indices, n_samples, replace=False))
    return selected_indices

def iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size, num_users, dataset_name='mnist'):
    train_users = {}
    test_users = {}

    if dataset_name == 'loop':
        train_idxs = np.arange(len(dataset_train))
        test_idxs = np.arange(len(dataset_test))

        for i in range(num_users):
            train_indices = get_samples(train_idxs, n_samples=dataset_train_size)
            test_indices = get_samples(test_idxs, n_samples=dataset_test_size)
            train_users[i] = train_indices
            test_users[i] = test_indices
        return train_users, test_users

    train_idxs = np.arange(len(dataset_train))
    train_labels = dataset_train.targets
    train_labels = np.vstack((train_idxs, train_labels))

    test_idxs = np.arange(len(dataset_test))
    test_labels = dataset_test.targets
    test_labels = np.vstack((test_idxs, test_labels))

    if dataset_name == 'mnist':
        data_classes = 10
    elif dataset_name == 'fashion_mnist':
        data_classes = 10
    elif dataset_name == 'cifar':
        data_classes = 10
    elif dataset_name == 'uci':
        data_classes = 6
    elif dataset_name == 'realworld':
        data_classes = 8
    else:
        data_classes = 0

    labels = list(range(data_classes))
    train_samples = int(dataset_train_size / data_classes)
    test_samples = int(dataset_test_size / data_classes)

    for i in range(num_users):
        train_indices = get_indices(train_labels, labels, n_samples=train_samples)
        test_indices = get_indices(test_labels, labels, n_samples=test_samples)
        train_users[i] = train_indices
        test_users[i] = test_indices
    return train_users, test_users