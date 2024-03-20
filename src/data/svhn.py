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


class SVHN_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, transform=None, target_transform=None, download=False, split = 'train'):
        self.root = root
        self.dataidxs = dataidxs
        self.train = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.split = self.train
        self.data, self.target = self.__build_truncated_dataset__()  # 调用类的私有方法来构建数据集

    def __build_truncated_dataset__(self): # 用于构建被截断后的数据集
        cifar_dataobj = datasets.SVHN(self.root, self.split, self.transform, self.target_transform, self.download)
        if self.train =='train': #检查是否挂载训练集
            data = cifar_dataobj.data
            data = data.transpose((0, 2, 3, 1))
            target = np.array(cifar_dataobj.labels)
        else:
            data = cifar_dataobj.data
            data = data.transpose((0, 2, 3, 1))
            target = np.array(cifar_dataobj.labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            # data = data.transpose((0, 2, 3, 1))
            target = target[self.dataidxs]
        return data, target   # 数据和标签是分开的，但是原来的是没有分开的

    def __getitem__(self, index): # 用于获取指定索引的样本
        img, target = Image.fromarray(self.data[index]), self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self): # 获取数据集的样本数量
        return len(self.data)


def  gen_data_split(data, num_users, num_classes, class_partitions):
    N = data.shape[0]
    data_class_idx = {i: np.where(data== i)[0] for i in range(num_classes)} # 得到每个类别中图像的索引  1:[xxx] 2:[xxxx]
    images_count_per_class = {i:len(data_class_idx[i]) for i in range(num_classes)} # 每个类别中图像的数量
    for data_idx in data_class_idx:
        np.random.shuffle(data_class_idx[data_idx]) # 随机打乱每个类别中图像的索引数组，以确保数据的随机性

    user_data_idx = collections.defaultdict(list) # 存储每个用户所包含的图像索引
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]): #对于每个用户，遍历其类别划分信息
            end_idx = int(images_count_per_class[c] * p)  # 计算每个类别中应该被分配给用户的图像数量，根据用户指定的概率,5000×概率。
            # end_idx = int(data_class_idx[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx]) # 将end_idx个数据集划分给usr_1
            data_class_idx[c] = data_class_idx[c][end_idx:]  # 去掉数据中已经分配过的
    for usr in user_data_idx: #再次循环遍历每个用户。
        np.random.shuffle(user_data_idx[usr])  # 对于每个用户，再次随机打乱其所包含的图像索引，以确保每个用户的数据是随机的。
    return user_data_idx




def load_SVHN_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = SVHN_truncated(datadir, download=True, transform=transform, split='train')
    cifar10_test_ds = SVHN_truncated(datadir, download=True, transform=transform,split='test')

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target




    return (X_train, y_train, X_test, y_test, cifar10_train_ds)




def partition_data(args):
    y_train = load_SVHN_data(args.dir_data)[1]  #取出训练标签
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


    labels = load_SVHN_data(args.dir_data)[1]
    n_classes = labels.max() + 1  # 10
    idx = [np.argwhere(np.array(labels) == y).flatten() for y in range(n_classes)]
    label_distribution = np.random.dirichlet([alpha]*num_users, n_classes)

    agent_dataid = {i: np.array([], dtype='int64') for i in range(num_users)}
    for c, fracs in zip(idx, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            agent_dataid[i] = np.concatenate((agent_dataid[i], idcs), axis=0)

    return agent_dataid

# 画图
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
    plt.savefig('figs/fenbu.png', dpi=500)
    # plt.show()





def get_agent_loader(args, kwargs):
    loaders_train = []
    if args.split == 'iid':
        agent_dataid = partition_data(args) #独立同分布
    else:
        agent_dataid = dirichlet_partition_data(args)

    data_train = load_SVHN_data(args.dir_data)[4]

    dataset_stats(agent_dataid, data_train, args)



    norm_mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    norm_std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    g = torch.Generator()
    g.manual_seed(0)
    if not args.test_only:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:# 用于控制是否进行随机水平翻转（数据增强操作）
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(transform_list)

    for i in range(args.n_agents):
        train_ds = SVHN_truncated(root=args.dir_data, dataidxs=agent_dataid[i], transform=transform_train, download=True, split='train')
        # print(train_ds)

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

    loader_train = None


    g = torch.Generator()
    g.manual_seed(0)
    if not args.test_only:  # 判断是否仅进行测试而不进行训练
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(transform_list)
        loader_train = DataLoader(
            datasets.SVHN(
                root=args.dir_data,
                split='train',
                download=True,
                transform=transform_train),
            batch_size=args.batch_size, shuffle=True,worker_init_fn=seed_worker,generator=g, **kwargs
        )


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

    loader_test = DataLoader(
        datasets.SVHN(
            root=args.dir_data,
            split='test',
            download=True,
            transform=transform_test),
        batch_size=700, shuffle=False, **kwargs
    )

    return loader_train, loader_test

