import os
import numpy as np
import scipy.io as sio

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms

class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path='', split='cross_rep'):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.seed     = seed
        self.split    = split   # 'cross_rep' (benchmark) or 'random' (within-rep 80/20)
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%d_%s_%s" %(self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.name += '_randsp' if split == 'random' else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()
        
    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%sData/%s' %(self.data_path, self.name)):
            # Get Raw data                
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trainset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                testset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=1)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trainset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
                
            if self.dataset == 'CIFAR100':
                print(self.dataset)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trainset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            
            if self.dataset == 'tinyimagenet':
                print(self.dataset)
                transform = transforms.Compose([# transforms.Resize(224),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], #pre-train
                                                #                      std=[0.229, 0.224, 0.225])])
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                                     std=[0.5, 0.5, 0.5])])
                # trainset = torchvision.datasets.ImageFolder(root='%sData/Raw' %self.data_path,
                #                                       train=True , download=True, transform=transform)
                # testset = torchvision.datasets.ImageFolder(root='%sData/Raw' %self.data_path,
                #                                       train=False, download=True, transform=transform)
                # root_dir = self.data_path
                root_dir = "./Data/Raw/tiny-imagenet-200/"
                trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
                trn_file = os.path.join(root_dir, 'train_list.txt')
                tst_file = os.path.join(root_dir, 'val_list.txt')
                with open(trn_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        trn_img_list.append(img)
                        trn_lbl_list.append(int(lbl))
                with open(tst_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        tst_img_list.append(img)
                        tst_lbl_list.append(int(lbl))
                trainset = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list, transformer=transform)
                testset = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list, transformer=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
            
            if self.dataset == 'ninapro':
                # ------------------------------------------------------------------ #
                # NinaPro DB2, Exercise B (file E1): 17 hand/wrist movements          #
                # Preprocessing: bandpass 10-500 Hz + rectification + z-score        #
                # Window: 400 samples = 200 ms @ 2000 Hz, stride 100 samples = 50 ms #
                # split='cross_rep': train reps {1,3,4,6}, test reps {2,5} (benchmark)
                # split='random':    all reps pooled, random 80/20 split by window   #
                # Rest windows discarded; labels 1-17 remapped to 0-16 -> n_cls=17   #
                # ------------------------------------------------------------------ #
                from scipy.signal import butter, filtfilt
                ninapro_path = os.path.join(self.data_path, 'Data', 'Raw', 'ninapro')
                WINDOW_SIZE  = 400    # 200 ms at 2000 Hz
                WINDOW_STEP  = 100    # 50 ms stride
                N_EMG_CH     = 12
                N_CLS        = 17
                FS           = 2000.0
                self.channels = 1; self.height = N_EMG_CH; self.width = WINDOW_SIZE; self.n_cls = N_CLS

                b_bp, a_bp = butter(4, [10.0 / (FS / 2), 500.0 / (FS / 2)], btype='band')

                TRAIN_REPS = {1, 3, 4, 6}
                TEST_REPS  = {2, 5}

                n_subjects = min(self.n_client, 40) if self.rule == 'subject' else 40

                per_subject_train_x = []
                per_subject_train_y = []
                test_x_list, test_y_list = [], []

                for subj in range(1, n_subjects + 1):
                    mat_path = os.path.join(ninapro_path, 'DB2_s%d' % subj, 'S%d_E1_A1.mat' % subj)
                    if not os.path.exists(mat_path):
                        mat_path = os.path.join(ninapro_path, 'DB2_s%d' % subj, 'DB2_s%d' % subj, 'S%d_E1_A1.mat' % subj)
                    mat    = sio.loadmat(mat_path)
                    emg    = mat['emg'].astype(np.float32)
                    labels = mat['restimulus'].flatten().astype(np.int64)
                    reps   = mat['rerepetition'].flatten().astype(np.int64)

                    for ch in range(N_EMG_CH):
                        emg[:, ch] = filtfilt(b_bp, a_bp, emg[:, ch])
                    emg = np.abs(emg)

                    if self.split == 'random':
                        # Use all repetitions; z-score on full recording
                        nz_mask = reps > 0
                        mean = emg[nz_mask].mean(axis=0, keepdims=True)
                        std  = emg[nz_mask].std(axis=0, keepdims=True) + 1e-6
                        emg  = (emg - mean) / std

                        # Window all reps (excluding rest label 0)
                        all_wx, all_wy = [], []
                        for s in range(0, len(emg) - WINDOW_SIZE + 1, WINDOW_STEP):
                            w_emg = emg[s:s + WINDOW_SIZE]
                            w_lbl = labels[s:s + WINDOW_SIZE]
                            counts  = np.bincount(w_lbl.astype(np.intp), minlength=18)
                            maj_lbl = int(np.argmax(counts))
                            if maj_lbl == 0 or counts[maj_lbl] / WINDOW_SIZE < 0.8:
                                continue
                            all_wx.append(w_emg.T[np.newaxis].copy())
                            all_wy.append(maj_lbl - 1)

                        if len(all_wx) == 0:
                            continue
                        all_wx = np.stack(all_wx, axis=0).astype(np.float32)
                        all_wy = np.array(all_wy, dtype=np.int64)

                        # Random 80/20 split per subject (stratified by class)
                        np.random.seed(self.seed + subj)
                        idx = np.arange(len(all_wy))
                        np.random.shuffle(idx)
                        n_train = int(0.8 * len(idx))
                        tr_idx, ts_idx = idx[:n_train], idx[n_train:]

                        per_subject_train_x.append(all_wx[tr_idx])
                        per_subject_train_y.append(all_wy[tr_idx])
                        test_x_list.append(all_wx[ts_idx])
                        test_y_list.append(all_wy[ts_idx])

                    else:  # cross_rep (benchmark standard)
                        tr_mask  = np.isin(reps, list(TRAIN_REPS))
                        tst_mask = np.isin(reps, list(TEST_REPS))
                        mean = emg[tr_mask].mean(axis=0, keepdims=True)
                        std  = emg[tr_mask].std(axis=0, keepdims=True) + 1e-6
                        emg  = (emg - mean) / std

                        for is_train, mask in [(True, tr_mask), (False, tst_mask)]:
                            seg_emg = emg[mask]
                            seg_lbl = labels[mask]
                            wx, wy  = [], []
                            for s in range(0, len(seg_emg) - WINDOW_SIZE + 1, WINDOW_STEP):
                                w_emg = seg_emg[s:s + WINDOW_SIZE]
                                w_lbl = seg_lbl[s:s + WINDOW_SIZE]
                                counts  = np.bincount(w_lbl.astype(np.intp), minlength=18)
                                maj_lbl = int(np.argmax(counts))
                                if maj_lbl == 0 or counts[maj_lbl] / WINDOW_SIZE < 0.8:
                                    continue
                                wx.append(w_emg.T[np.newaxis].copy())
                                wy.append(maj_lbl - 1)
                            if len(wx) > 0:
                                wx_arr = np.stack(wx, axis=0).astype(np.float32)
                                wy_arr = np.array(wy, dtype=np.int64)
                                if is_train:
                                    per_subject_train_x.append(wx_arr)
                                    per_subject_train_y.append(wy_arr)
                                else:
                                    test_x_list.append(wx_arr)
                                    test_y_list.append(wy_arr)

                test_x  = np.concatenate(test_x_list, axis=0).reshape(-1, 1, N_EMG_CH, WINDOW_SIZE)
                test_y  = np.concatenate(test_y_list).reshape(-1, 1).astype(np.int64)
                train_x = np.concatenate(per_subject_train_x, axis=0)
                train_y = np.concatenate(per_subject_train_y).reshape(-1, 1).astype(np.int64)
                print('NinaPro Ex-B loaded: %d train windows, %d test windows, %d subjects, %d classes (split=%s)'
                      % (len(train_x), len(test_x), n_subjects, N_CLS, self.split))

            if self.dataset not in ['emnist', 'ninapro']:
                train_itr = train_load.__iter__(); test_itr = test_load.__iter__()
                # labels are of shape (n_data,)
                train_x, train_y = train_itr.__next__()
                test_x, test_y = test_itr.__next__()

                train_x = train_x.numpy(); train_y = train_y.numpy().reshape(-1,1)
                test_x = test_x.numpy(); test_y = test_y.numpy().reshape(-1,1)
            
            
            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "Data/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0

                # take first 10 classes of letters
                train_idx = np.where(y_train < 10)[0]

                y_train = y_train[train_idx]
                x_train = x_train[train_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0

                test_idx = np.where(y_test < 10)[0]

                y_test = y_test[test_idx]
                x_test = x_test[test_idx]
                
                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))
                
                # normalise train and test features

                train_x = (x_train - mean_x) / std_x
                train_y = y_train
                
                test_x = (x_test  - mean_x) / std_x
                test_y = y_test
                
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]
            
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y
            
            
            ###
            n_data_per_client = int((len(train_y)) / self.n_client)
            # Draw from lognormal distribution
            # client_data_list = (np.random.lognormal(mean=np.log(n_data_per_client), sigma=self.unbalanced_sgm, size=self.n_client))
            # client_data_list = (client_data_list/(np.sum(client_data_list)/len(train_y)))
            client_data_list = np.ones(self.n_client, dtype=int)*n_data_per_client
            diff = np.sum(client_data_list) - len(train_y)
            
            # Add/Subtract the excess number starting from first client
            if diff!= 0:
                for client_i in range(self.n_client):
                    if client_data_list[client_i] > diff:
                        client_data_list[client_i] -= diff
                        break
            ###     
            
            if self.rule == 'Dirichlet' or self.rule == 'Pathological':
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                    # np.save("results/heterogeneity_distribution_{:s}.npy".format(self.dataset), cls_priors)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client,self.n_cls])
                    a[:,c::] = 0
                    [np.random.shuffle(i) for i in a]
                    # np.save("results/heterogeneity_distribution_{:s}_{:s}.npy".format(self.dataset, self.rule), a/c)
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i,j] != 0:
                                prior_cumsum[i,j] = a[i,0:j+1].sum()/c*1.0

                idx_list = [np.where(train_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                # print(cls_amount)
                client_x = [ np.zeros((client_data_list[client__], self.channels, self.height, self.width)).astype(np.float32) for client__ in range(self.n_client) ]
                client_y = [ np.zeros((client_data_list[client__], 1)).astype(np.int64) for client__ in range(self.n_client) ]
    
                while(np.sum(client_data_list)!=0):
                    curr_client = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    # print('Remaining Data: %d' %np.sum(client_data_list))
                    if client_data_list[curr_client] <= 0:
                        continue
                    client_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if train_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            cls_amount [cls_label] = len(idx_list[cls_label]) 
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1
                        
                        client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                print(true_sample)
                client_x = np.asarray(client_x)
                client_y = np.asarray(client_y)
                
                # cls_means = np.zeros((self.n_client, self.n_cls))
                # for client in range(self.n_client):
                #     for cls in range(self.n_cls):
                #         cls_means[client,cls] = np.mean(client_y[client]==cls)
                # prior_real_diff = np.abs(cls_means-cls_priors)
                # print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                # print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(train_y)//100 % self.n_client == 0 
                
                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(train_y[:, 0])
                n_data_per_client = len(train_y) // self.n_client
                # client_x dtype needs to be float32, the same as weights
                client_x = np.zeros((self.n_client, n_data_per_client, 3, 32, 32), dtype=np.float32)
                client_y = np.zeros((self.n_client, n_data_per_client, 1), dtype=np.float32)
                train_x = train_x[idx] # 50000*3*32*32
                train_y = train_y[idx]
                n_cls_sample_per_device = n_data_per_client // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        client_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = train_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        client_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = train_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :] 
            
            
            elif self.rule == 'iid':
                # Truncate to exact multiple of n_client so all clients get equal sizes
                n_keep = (len(train_y) // self.n_client) * self.n_client
                train_x = train_x[:n_keep]
                train_y = train_y[:n_keep]
                n_data_per_client = n_keep // self.n_client

                client_x = train_x.reshape(self.n_client, n_data_per_client, *train_x.shape[1:])
                client_y = train_y.reshape(self.n_client, n_data_per_client, *train_y.shape[1:])

            elif self.rule == 'subject':
                # Each NinaPro subject becomes one FL client.
                # Subsample to the minimum window count so all clients are balanced.
                if self.dataset != 'ninapro':
                    raise ValueError("rule='subject' is only supported for dataset='ninapro'")
                min_win = min(len(x) for x in per_subject_train_x)
                np.random.seed(self.seed)
                client_x_list, client_y_list = [], []
                for i in range(len(per_subject_train_x)):
                    idx = np.random.choice(len(per_subject_train_x[i]), min_win, replace=False)
                    client_x_list.append(per_subject_train_x[i][idx])
                    client_y_list.append(per_subject_train_y[i][idx].reshape(-1, 1))
                client_x = np.stack(client_x_list, axis=0)  # (n_client, min_win, 1, 12, 200)
                client_y = np.stack(client_y_list, axis=0)  # (n_client, min_win, 1)

            self.client_x = client_x; self.client_y = client_y

            self.test_x  = test_x;  self.test_y  = test_y
            
            # Save data
            print('begin to save data...')
            os.mkdir('%sData/%s' %(self.data_path, self.name))
            
            np.save('%sData/%s/client_x.npy' %(self.data_path, self.name), client_x)
            np.save('%sData/%s/client_y.npy' %(self.data_path, self.name), client_y)

            np.save('%sData/%s/test_x.npy'  %(self.data_path, self.name),  test_x)
            np.save('%sData/%s/test_y.npy'  %(self.data_path, self.name),  test_y)

            print('data loading finished.')

        else:
            print("Data is already downloaded")
            self.client_x = np.load('%sData/%s/client_x.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.client_y = np.load('%sData/%s/client_y.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.n_client = len(self.client_x)

            self.test_x  = np.load('%sData/%s/test_x.npy'  %(self.data_path, self.name), mmap_mode = 'r')
            self.test_y  = np.load('%sData/%s/test_y.npy'  %(self.data_path, self.name), mmap_mode = 'r')
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'tinyimagenet':
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
            if self.dataset == 'ninapro':
                self.channels = 1; self.width = 400; self.height = 12; self.n_cls = 17;
            
            print('data loading finished.')
                
        '''
        print('Class frequencies:')
        count = 0
        for client in range(self.n_client):
            print("Client %3d: " %client + 
                  ', '.join(["%.3f" %np.mean(self.client_y[client]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.client_y[client].shape[0])
            count += self.client_y[client].shape[0]
        
        
        print('Total Amount:%d' %count)
        print('--------')

        print("      Test: " + 
              ', '.join(["%.3f" %np.mean(self.test_y==cls) for cls in range(self.n_cls)]) + 
              ', Amount:%d' %self.test_y.shape[0])
        '''
        
def generate_syn_logistic(dimension, n_client, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False, iid_dat=False):
    
    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points
    
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)
    
    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_client)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' %np.sum(samples_per_user))
    
    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_client))
    data_y = list(range(n_client))

    mean_W = np.random.normal(0, alpha, n_client)
    B = np.random.normal(0, beta, n_client)

    mean_x = np.zeros((n_client, dimension))

    if not iid_dat: # If IID then make all 0s.
        for i in range(n_client):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))
    
    if iid_sol: # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))
    
    for i in range(n_client):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1,1)

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
            # data_x: numpy (N, 1, 12, 200) float32
            # data_y: numpy (N, 1) int64  or  bool
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
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img # Horizontal flip
                if (np.random.rand() > .5):
                # Random cropping 
                    pad = 4
                    extended_img = np.zeros((3,32 + pad *2, 32 + pad *2)).astype(np.float32)
                    extended_img[:,pad:-pad,pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:,dim_1:dim_1+32,dim_2:dim_2+32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
                
        elif self.name == 'tinyimagenet':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if np.random.rand() > .5:
                    # Random cropping
                    pad = 8
                    extended_img = np.zeros((3, 64 + pad * 2, 64 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 64, dim_2:dim_2 + 64]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == 'ninapro':
            X = torch.tensor(self.X_data[idx]).float()   # (1, 12, 200)
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
        # ********************
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list) 
