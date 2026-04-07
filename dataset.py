import os
import numpy as np
import scipy.io as sio
import scipy.io as io

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path='', split='cross_rep'):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.seed     = seed
        self.split    = split
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%d_%s_%s" % (self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' % unbalanced_sgm if unbalanced_sgm != 0 else ''
        self.name += '_randsp' if split == 'random' else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()

    def set_data(self):
        if not os.path.exists('%sData/%s' % (self.data_path, self.name)):
            # ------------------------------------------------------------------
            # Standard datasets
            # ------------------------------------------------------------------
            if self.dataset == 'mnist':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                trainset = torchvision.datasets.MNIST(
                    root='%sData/Raw' % self.data_path,
                    train=True, download=True, transform=transform
                )
                testset = torchvision.datasets.MNIST(
                    root='%sData/Raw' % self.data_path,
                    train=False, download=True, transform=transform
                )

                train_load = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=1)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10

            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                ])

                trainset = torchvision.datasets.CIFAR10(
                    root='%sData/Raw' % self.data_path,
                    train=True, download=True, transform=transform
                )
                testset = torchvision.datasets.CIFAR10(
                    root='%sData/Raw' % self.data_path,
                    train=False, download=True, transform=transform
                )

                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 10

            if self.dataset == 'CIFAR100':
                print(self.dataset)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
                ])
                trainset = torchvision.datasets.CIFAR100(
                    root='%sData/Raw' % self.data_path,
                    train=True, download=True, transform=transform
                )
                testset = torchvision.datasets.CIFAR100(
                    root='%sData/Raw' % self.data_path,
                    train=False, download=True, transform=transform
                )
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3
                self.width = 32
                self.height = 32
                self.n_cls = 100

            if self.dataset == 'tinyimagenet':
                print(self.dataset)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                root_dir = "./Data/Raw/tiny-imagenet-200/"
                trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []

                trn_file = os.path.join(root_dir, 'train_list.txt')
                tst_file = os.path.join(root_dir, 'val_list.txt')

                with open(trn_file) as f:
                    for line in f.readlines():
                        img, lbl = line.strip().split()
                        trn_img_list.append(img)
                        trn_lbl_list.append(int(lbl))

                with open(tst_file) as f:
                    for line in f.readlines():
                        img, lbl = line.strip().split()
                        tst_img_list.append(img)
                        tst_lbl_list.append(int(lbl))

                trainset = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list, transformer=transform)
                testset = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list, transformer=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
                self.channels = 3
                self.width = 64
                self.height = 64
                self.n_cls = 200

            # ------------------------------------------------------------------
            # NinaPro DB2 Exercise C (grasp-based)
            # ------------------------------------------------------------------
            if self.dataset == 'ninapro':
                from scipy.signal import butter, filtfilt

                # NinaPro DB2 / Exercise C
                ninapro_path = os.path.join(self.data_path, 'Data', 'Raw', 'ninapro')
                WINDOW_SIZE  = 400    # 200 ms @ 2000 Hz
                WINDOW_STEP  = 100    # 50 ms stride
                N_EMG_CH     = 12
                N_CLS        = 8
                FS           = 2000.0

                self.channels = 1
                self.height   = N_EMG_CH
                self.width    = WINDOW_SIZE
                self.n_cls    = N_CLS

                # Your selected Exercise C gestures
                # 1  = Large diameter
                # 20 = Power disk
                # 18 = Parallel extension grasp (used for your prismatic-extension choice)
                # 3  = Fixed hook grasp
                # 15 = Tip pinch grasp
                # 12 = Precision sphere grasp
                # 13 = Tripod grasp
                # 9  = Writing tripod grasp
                SELECTED_GESTURES = [1, 20, 18, 3, 15, 12, 13, 9]
                label_map = {g: i for i, g in enumerate(SELECTED_GESTURES)}

                # Your chosen representative subjects
                FEMALE_SUBJECTS = [38, 28, 22, 19, 14]
                MALE_SUBJECTS   = [40, 3, 15, 21, 29]

                # Balanced split: 8 train / 2 test
                TRAIN_SUBJECTS = [38, 28, 22, 19, 40, 3, 15, 21]
                TEST_SUBJECTS  = [14, 29]
                SELECTED_SUBJECTS = TRAIN_SUBJECTS + TEST_SUBJECTS

                b_bp, a_bp = butter(4, [10.0 / (FS / 2), 500.0 / (FS / 2)], btype='band')

                per_subject_train_x = []
                per_subject_train_y = []
                test_x_list, test_y_list = [], []

                for subj in SELECTED_SUBJECTS:
                    mat_path = os.path.join(ninapro_path, f'DB2_s{subj}', f'S{subj}_E2_A1.mat')
                    if not os.path.exists(mat_path):
                        mat_path = os.path.join(ninapro_path, f'DB2_s{subj}', f'DB2_s{subj}', f'S{subj}_E2_A1.mat')

                    if not os.path.exists(mat_path):
                        raise FileNotFoundError(f"Could not find NinaPro file for subject {subj}: {mat_path}")

                    mat    = sio.loadmat(mat_path)
                    emg    = mat['emg'].astype(np.float32)
                    labels = mat['restimulus'].flatten().astype(np.int64)
                    reps   = mat['rerepetition'].flatten().astype(np.int64)

                    # Bandpass filter + rectification
                    for ch in range(N_EMG_CH):
                        emg[:, ch] = filtfilt(b_bp, a_bp, emg[:, ch])
                    emg = np.abs(emg)

                    # Subject-wise normalization using non-rest samples
                    nz_mask = reps > 0
                    mean = emg[nz_mask].mean(axis=0, keepdims=True)
                    std  = emg[nz_mask].std(axis=0, keepdims=True) + 1e-6
                    emg  = (emg - mean) / std

                    wx, wy = [], []

                    for s in range(0, len(emg) - WINDOW_SIZE + 1, WINDOW_STEP):
                        w_emg = emg[s:s + WINDOW_SIZE]
                        w_lbl = labels[s:s + WINDOW_SIZE]

                        counts  = np.bincount(w_lbl.astype(np.intp), minlength=50)
                        maj_lbl = int(np.argmax(counts))

                        # Keep only clean windows from selected gestures
                        if maj_lbl not in SELECTED_GESTURES:
                            continue
                        if counts[maj_lbl] / WINDOW_SIZE < 0.8:
                            continue

                        wx.append(w_emg.T[np.newaxis].copy())  # (1, 12, 400)
                        wy.append(label_map[maj_lbl])          # 0..7

                    if len(wx) == 0:
                        continue

                    wx_arr = np.stack(wx, axis=0).astype(np.float32)
                    wy_arr = np.array(wy, dtype=np.int64)

                    if subj in TRAIN_SUBJECTS:
                        per_subject_train_x.append(wx_arr)
                        per_subject_train_y.append(wy_arr)
                    elif subj in TEST_SUBJECTS:
                        test_x_list.append(wx_arr)
                        test_y_list.append(wy_arr)

                if len(per_subject_train_x) == 0 or len(test_x_list) == 0:
                    raise RuntimeError("No windows produced for NinaPro. Check subjects, labels, or file paths.")

                train_x = np.concatenate(per_subject_train_x, axis=0)
                train_y = np.concatenate(per_subject_train_y).reshape(-1, 1).astype(np.int64)

                test_x  = np.concatenate(test_x_list, axis=0).reshape(-1, 1, N_EMG_CH, WINDOW_SIZE)
                test_y  = np.concatenate(test_y_list).reshape(-1, 1).astype(np.int64)

                print('NinaPro Ex-C loaded: %d train windows, %d test windows, %d selected subjects, %d classes'
                      % (len(train_x), len(test_x), len(SELECTED_SUBJECTS), N_CLS))

            # ------------------------------------------------------------------
            # Standard extraction for non-NinaPro datasets
            # ------------------------------------------------------------------
            if self.dataset not in ['emnist', 'ninapro']:
                train_itr = train_load.__iter__()
                test_itr = test_load.__iter__()
                train_x, train_y = train_itr.__next__()
                test_x, test_y = test_itr.__next__()

                train_x = train_x.numpy()
                train_y = train_y.numpy().reshape(-1, 1)
                test_x  = test_x.numpy()
                test_y  = test_y.numpy().reshape(-1, 1)

            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "Data/Raw/matlab/emnist-letters.mat")
                x_train = emnist["dataset"][0][0][0][0][0][0].astype(np.float32)
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1

                train_idx = np.where(y_train < 10)[0]
                y_train = y_train[train_idx]
                x_train = x_train[train_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                x_test = emnist["dataset"][0][0][1][0][0][0].astype(np.float32)
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1

                test_idx = np.where(y_test < 10)[0]
                y_test = y_test[test_idx]
                x_test = x_test[test_idx]

                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))

                train_x = (x_train - mean_x) / std_x
                train_y = y_train
                test_x  = (x_test - mean_x) / std_x
                test_y  = y_test

                self.channels = 1
                self.width = 28
                self.height = 28
                self.n_cls = 10

            # Shuffle training data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]

            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y

            # ------------------------------------------------------------------
            # Client partitioning
            # ------------------------------------------------------------------
            n_data_per_client = int((len(train_y)) / self.n_client)
            client_data_list = np.ones(self.n_client, dtype=int) * n_data_per_client
            diff = np.sum(client_data_list) - len(train_y)

            if diff != 0:
                for client_i in range(self.n_client):
                    if client_data_list[client_i] > diff:
                        client_data_list[client_i] -= diff
                        break

            if self.rule == 'Dirichlet' or self.rule == 'Pathological':
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg] * self.n_cls, size=self.n_client)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client, self.n_cls])
                    a[:, c::] = 0
                    [np.random.shuffle(i) for i in a]
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i, j] != 0:
                                prior_cumsum[i, j] = a[i, 0:j+1].sum() / c * 1.0

                idx_list = [np.where(train_y == i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for _ in range(self.n_cls)]

                client_x = [
                    np.zeros((client_data_list[client__], self.channels, self.height, self.width)).astype(np.float32)
                    for client__ in range(self.n_client)
                ]
                client_y = [
                    np.zeros((client_data_list[client__], 1)).astype(np.int64)
                    for client__ in range(self.n_client)
                ]

                while np.sum(client_data_list) != 0:
                    curr_client = np.random.randint(self.n_client)
                    if client_data_list[curr_client] <= 0:
                        continue
                    client_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        if cls_amount[cls_label] <= 0:
                            cls_amount[cls_label] = len(idx_list[cls_label])
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1

                        client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]
                        break

                print(true_sample)
                client_x = np.asarray(client_x)
                client_y = np.asarray(client_y)

            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm == 0:
                assert len(train_y) // 100 % self.n_client == 0

                idx = np.argsort(train_y[:, 0])
                n_data_per_client = len(train_y) // self.n_client
                client_x = np.zeros((self.n_client, n_data_per_client, 3, 32, 32), dtype=np.float32)
                client_y = np.zeros((self.n_client, n_data_per_client, 1), dtype=np.float32)
                train_x = train_x[idx]
                train_y = train_y[idx]
                n_cls_sample_per_device = n_data_per_client // 100
                for i in range(self.n_client):
                    for j in range(100):
                        client_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = \
                            train_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        client_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = \
                            train_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :]

            elif self.rule == 'iid':
                n_keep = (len(train_y) // self.n_client) * self.n_client
                train_x = train_x[:n_keep]
                train_y = train_y[:n_keep]
                n_data_per_client = n_keep // self.n_client

                client_x = train_x.reshape(self.n_client, n_data_per_client, *train_x.shape[1:])
                client_y = train_y.reshape(self.n_client, n_data_per_client, *train_y.shape[1:])

            elif self.rule == 'subject':
                if self.dataset != 'ninapro':
                    raise ValueError("rule='subject' is only supported for dataset='ninapro'")

                min_win = min(len(x) for x in per_subject_train_x)
                np.random.seed(self.seed)
                client_x_list, client_y_list = [], []

                for i in range(len(per_subject_train_x)):
                    idx = np.random.choice(len(per_subject_train_x[i]), min_win, replace=False)
                    client_x_list.append(per_subject_train_x[i][idx])
                    client_y_list.append(per_subject_train_y[i][idx].reshape(-1, 1))

                client_x = np.stack(client_x_list, axis=0)  # (n_train_subjects, min_win, 1, 12, 400)
                client_y = np.stack(client_y_list, axis=0)  # (n_train_subjects, min_win, 1)

            self.client_x = client_x
            self.client_y = client_y
            self.test_x = test_x
            self.test_y = test_y

            print('begin to save data...')
            os.makedirs('%sData/%s' % (self.data_path, self.name), exist_ok=True)

            np.save('%sData/%s/client_x.npy' % (self.data_path, self.name), client_x)
            np.save('%sData/%s/client_y.npy' % (self.data_path, self.name), client_y)
            np.save('%sData/%s/test_x.npy' % (self.data_path, self.name), test_x)
            np.save('%sData/%s/test_y.npy' % (self.data_path, self.name), test_y)

            print('data loading finished.')

        else:
            print("Data is already downloaded")
            self.client_x = np.load('%sData/%s/client_x.npy' % (self.data_path, self.name), mmap_mode='r')
            self.client_y = np.load('%sData/%s/client_y.npy' % (self.data_path, self.name), mmap_mode='r')
            self.n_client = len(self.client_x)

            self.test_x = np.load('%sData/%s/test_x.npy' % (self.data_path, self.name), mmap_mode='r')
            self.test_y = np.load('%sData/%s/test_y.npy' % (self.data_path, self.name), mmap_mode='r')

            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10
            if self.dataset == 'tinyimagenet':
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200
            if self.dataset == 'ninapro':
                self.channels = 1; self.width = 400; self.height = 12; self.n_cls = 8

            print('data loading finished.')


def generate_syn_logistic(dimension, n_client, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False, iid_dat=False):
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_client)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' % np.sum(samples_per_user))

    data_x = list(range(n_client))
    data_y = list(range(n_client))

    mean_W = np.random.normal(0, alpha, n_client)
    B = np.random.normal(0, beta, n_client)

    mean_x = np.zeros((n_client, dimension))

    if not iid_dat:
        for i in range(n_client):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))

    if iid_sol:
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))

    for i in range(n_client):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1, 1)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name

        if self.name == 'mnist' or self.name == 'emnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == "tinyimagenet":
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')

        elif self.name == 'ninapro':
            # data_x: numpy (N, 1, 12, 400) float32
            # data_y: numpy (N, 1) int64 or bool
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'emnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            y = self.y_data[idx]
            return X, y

        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img
                if np.random.rand() > .5:
                    pad = 4
                    extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1+32, dim_2:dim_2+32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            y = self.y_data[idx]
            return img, y

        elif self.name == 'tinyimagenet':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img
                if np.random.rand() > .5:
                    pad = 8
                    extended_img = np.zeros((3, 64 + pad * 2, 64 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 64, dim_2:dim_2 + 64]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            y = self.y_data[idx]
            return img, y

        elif self.name == 'ninapro':
            X = torch.tensor(self.X_data[idx]).float()   # (1, 12, 400)
            if isinstance(self.y_data, bool):
                return X
            y = self.y_data[idx]
            return X, y

        else:
            raise NotImplementedError


class DatasetFromDir(data.Dataset):
    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)