import torch
import numpy as np
from scipy.special import comb
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from utils.visual import *

def fps(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def idg(targets, data, cfg):
    def binarize_class(y):
        label = y.reshape(len(y), -1)
        enc = OneHotEncoder(categories='auto')
        enc.fit(label)
        label = enc.transform(label).toarray().astype(np.float32)
        label = torch.from_numpy(label)
        return label

    def create_model(ds, feature, c):
        from models.partial_models.resnet import resnet
        from models.partial_models.mlp import mlp_phi
        from models.partial_models.wide_resnet import WideResNet
        if ds in ['kmnist', 'fmnist']:
            model = mlp_phi(feature, c)
        elif ds in ['CIFAR10']:
            model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        elif ds in ['CIFAR100']:
            model = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
        else:
            pass
        return model

    with torch.no_grad():
        c = max(targets) + 1
        data = torch.from_numpy(data)
        y = binarize_class(targets.clone().detach().long())
        ds = cfg['dataset']

        f = np.prod(list(data.shape)[1:])
        batch_size = 2000
        rate = 0.4
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight_path = f'./weights/{ds}.pt'
            
        model = create_model(ds, f, c).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        train_X, train_Y = data.to(device), y.to(device)

        train_X = train_X.permute(0, 3, 1, 2).to(torch.float32)
        train_p_Y_list = []
        step = train_X.size(0) // batch_size
        for i in range(0, step):
            outputs = model(train_X[i * batch_size:(i + 1) * batch_size])
            train_p_Y = train_Y[i * batch_size:(i + 1) * batch_size].clone().detach()
            partial_rate_array = F.softmax(outputs, dim=1).clone().detach()
            partial_rate_array[torch.where(train_Y[i * batch_size:(i + 1) * batch_size] == 1)] = 0
            partial_rate_array = partial_rate_array / torch.max(partial_rate_array, dim=1, keepdim=True)[0]
            partial_rate_array = partial_rate_array / partial_rate_array.mean(dim=1, keepdim=True) * rate
            partial_rate_array[partial_rate_array > 1.0] = 1.0
            m = torch.distributions.binomial.Binomial(total_count=1, probs=partial_rate_array)
            z = m.sample()
            train_p_Y[torch.where(z == 1)] = 1.0
            train_p_Y_list.append(train_p_Y)
        train_p_Y = torch.cat(train_p_Y_list, dim=0)
        assert train_p_Y.shape[0] == train_X.shape[0]
    final_y = train_p_Y.cpu().clone()
    pn = final_y.sum() / torch.ones_like(final_y).sum()
    print("Partial type: instance dependent, Average Label: " + str(pn * 10))
    return train_p_Y.cpu()


def uss(train_labels): 
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1
        
    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]
    cardinality = (2**K - 2).float()
    number = torch.tensor([comb(K, i+1) for i in range(K-1)]).float() # 1 to K-1 because cannot be empty or full label set, convert list to tensor
    frequency_dis = number / cardinality
    prob_dis = torch.zeros(K-1) # tensor of K-1
    for i in range(K-1):
        if i == 0:
            prob_dis[i] = frequency_dis[i]
        else:
            prob_dis[i] = frequency_dis[i]+prob_dis[i-1]

    random_n = torch.from_numpy(np.random.uniform(0, 1, n)).float() # tensor: n
    mask_n = torch.ones(n) # n is the number of train_data
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    
    temp_num_partial_train_labels = 0 # save temp number of partial train_labels
    
    for j in range(n): # for each instance
        for jj in range(K-1): # 0 to K-2
            if random_n[j] <= prob_dis[jj] and mask_n[j] == 1:
                temp_num_partial_train_labels = jj+1 # decide the number of partial train_labels
                mask_n[j] = 0
                
        temp_num_fp_train_labels = temp_num_partial_train_labels - 1
        candidates = torch.from_numpy(np.random.permutation(K.item())).long() # because K is tensor type
        candidates = candidates[candidates!=train_labels[j]]
        temp_fp_train_labels = candidates[:temp_num_fp_train_labels]
        
        partialY[j, temp_fp_train_labels] = 1.0 # fulfill the partial label matrix
    print("Finish Generating Candidate Label Sets!\n")
    return partialY



def pre_filter(cfg, partialY, data, labels):
    print('Average candidate num: ', partialY.sum(1).mean())
    
    mask = torch.zeros_like(partialY)
    top_k_percent_indices = torch.topk(cfg['zsclip'], int(mask.shape[1] * 0.5), dim=1)[1]

    for j, indices in enumerate(top_k_percent_indices):
        mask[j, indices] = 1
    
    partialY = partialY * mask
    
    visualize(partialY, labels)
    partialY, data, labels = remove_zero_rows(partialY, data, labels)
    visualize(partialY, labels)
    
    return partialY, data, labels