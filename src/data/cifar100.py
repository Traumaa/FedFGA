from email import generator
from importlib import import_module
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import torch
import numpy as np
import collections
from PIL import Image

from collections import Counter
import matplotlib.pyplot as plt

class CIFAR100_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = datasets.CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
        if self.train:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = Image.fromarray(self.data[index]), self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def gen_data_split(data, num_users, num_classes, class_partitions):
    N = data.shape[0]
    data_class_idx = {i: np.where(data== i)[0] for i in range(num_classes)}
    images_count_per_class = {i:len(data_class_idx[i]) for i in range(num_classes)}
    for data_idx in data_class_idx:
        np.random.shuffle(data_class_idx[data_idx])

    user_data_idx = collections.defaultdict(list)
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(images_count_per_class[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]
    for usr in user_data_idx:
        np.random.shuffle(user_data_idx[usr])
    return user_data_idx

def partition_data(args):
    y_train = load_cifar10_data(args.dir_data)[1]  #取出训练标签
    num_classes = 10
    classes_per_user = 10 if args.split=='iid' else 3  # 根据args.split的值来确定每个智能体（agent）分配的类别数。如果args.split为'iid'，则每个客户端分配10个类别，否则分配3个类别。
    num_users = args.n_agents

    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"  # 判断每个（agent）分配的类别数量满足要求的
    count_per_class = (classes_per_user * num_users) // num_classes  # 10*10/10=10  3*10/10=3
    class_dict = {}
    for i in range(num_classes):  # 给十个客户端分配类别概率
        probs = np.random.uniform(1, 1, size=count_per_class) # 111...
        probs_norm = (probs / probs.sum()).tolist() # 进行归一化，将数组转换为列表
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
    # {0:{count:10, prob=[0.1,0.1,0.1.......]}}
    class_partitions = collections.defaultdict(list)
    for i in range(num_users): # 给每个客户端分配数据类别id
        c = [] # 存储当前用户选择的类别
        for _ in range(classes_per_user): #10 或者3  选择三个类别加入客户端
            class_counts = [class_dict[i]['count'] for i in range(num_classes)] # 数据类别计数[10,10,10,10...]  或者[3，3，3，3...]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0] #0
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1  # class_dict会逐渐减少
        class_partitions['class'].append(c) #
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c]) # 字典里包含类别和概率

    agent_dataid = gen_data_split(y_train, num_users, num_classes, class_partitions)
    # agent_dataid = agent_dataid # 字典里包含客户端和数据id

    return agent_dataid

def dirichlet_partition_data(args):
    alpha = args.alpha
    num_users = args.n_agents


    labels = load_cifar100_data(args.dir_data)[1]
    n_classes = labels.max() + 1  # 10
    idx = [np.argwhere(np.array(labels) == y).flatten() for y in range(n_classes)]
    label_distribution = np.random.dirichlet([alpha]*num_users, n_classes)

    agent_dataid = {i: np.array([], dtype='int64') for i in range(num_users)}
    for c, fracs in zip(idx, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            agent_dataid[i] = np.concatenate((agent_dataid[i], idcs), axis=0)

    return agent_dataid

def dataset_stats(dict_users, dataset, args):
    # dict users {0: array([], dtype=int64), 1: array([], dtype=int64), ..., 100: array([], dtype=int64)}
    num_classes = 10
    stats = {i: np.array([], dtype='int64') for i in range(len(dict_users))}
    for key, value in dict_users.items():
        for x in value:
            stats[key] = np.concatenate((stats[key], np.array([dataset[x][1]])), axis=0)  #dataset[x][1]]是数据x的标签

    nparray = np.zeros([num_classes, args.n_agents], dtype=int)
    for j in range(args.n_agents):
        cls = stats[j]
        cls_counter = Counter(cls)
        for i in range(num_classes):
            nparray[i][j] = cls_counter[i]

    fig, ax = plt.subplots()
    bottom = np.zeros([args.n_agents], dtype=int)
    for cls in range(num_classes):
        ax.bar(range(args.n_agents), nparray[cls], bottom=bottom, label='class{}'.format(cls))
        bottom += nparray[cls]
    ax.legend(loc='lower right')
    plt.title('Data Distribution')
    plt.xlabel('Clients')
    plt.ylabel('Amount of Training Data')
    plt.savefig('figs/cifar100_fenbu_iid.png', dpi=500)
    # plt.show()

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test, cifar10_train_ds)



def partition_data(args):    
    y_train = load_cifar100_data(args.dir_data)[1]
    num_classes = 100
    classes_per_user = 100 if args.split=='iid' else 30
    num_users = args.n_agents
    
    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        probs = np.random.uniform(1, 1, size=count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
    # {0:{count:10, prob=[0.1,0.1,0.1.......]}}
    class_partitions = collections.defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user): #10
            class_counts = [class_dict[i]['count'] for i in range(num_classes)] #[10,10,10,10...]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0] #0
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
        
    agent_dataid = gen_data_split(y_train, num_users, num_classes, class_partitions)

    return agent_dataid

def get_agent_loader(args, kwargs):
    loaders_train = []
    # agent_dataid = partition_data(args) #dict
    if args.split == 'iid':
        agent_dataid = partition_data(args) #独立同分布
    else:
        agent_dataid = dirichlet_partition_data(args)

    data_train = load_cifar100_data(args.dir_data)[4]
    dataset_stats(agent_dataid, data_train, args)

    norm_mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    norm_std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    g = torch.Generator()
    g.manual_seed(0)
    if not args.test_only:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(transform_list)

    for i in range(args.n_agents):
        train_ds = CIFAR100_truncated(root=args.dir_data, dataidxs=agent_dataid[i], train=True, transform=transform_train, download=True)
        train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker,generator=g, **kwargs)
        loaders_train.append(train_dl)
    
    return loaders_train

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def get_loader(args, kwargs):
    norm_mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    norm_std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    #norm_mean = [0.49139968, 0.48215827, 0.44653124]
    #norm_std = [0.24703233, 0.24348505, 0.26158768]
    loader_train = None

    if not args.test_only:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        loader_train = DataLoader(
            datasets.CIFAR100(
                root=args.dir_data,
                train=True,
                download=True,
                transform=transform_train),
            batch_size=args.batch_size * args.n_GPUs, shuffle=True, **kwargs
        )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    loader_test = DataLoader(
        datasets.CIFAR100(
            root=args.dir_data,
            train=False,
            download=True,
            transform=transform_test),
        batch_size=256 * args.n_GPUs, shuffle=False, **kwargs
    )

    return loader_train, loader_test
