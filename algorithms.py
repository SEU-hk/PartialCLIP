import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import math
import copy
import numpy as np
import random
import scipy.sparse as sp
import os
from sklearn.metrics import euclidean_distances

ALGORITHMS = [
    'PRODEN',
    'CAVL',
    'ABS_MAE',
    'ABS_GCE',
    'CC',
    'LWS',
    'CRDPLL',
    'POP',
    'IDGP',    
    'ABLE',
    'Solar',
    'HTC',
    'RECORDS',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a partial-label learning algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, model, input_shape, train_givenY, hparams):
        super(Algorithm, self).__init__()
        self.network = model
        self.hparams = hparams
        self.num_data = input_shape[0]
        self.num_classes = train_givenY.shape[1]

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class PRODEN(Algorithm):
    """
    PRODEN
    Reference: Progressive identification of true labels for partial-label learning, ICML 2020.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(PRODEN, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float()/tempY
        self.label_confidence = label_confidence

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        with autocast():
            loss = self.rc_loss(self.predict(x), index)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.confidence_update(x, partial_y, index)
            return loss
            # return {'loss': loss.item()}

    def rc_loss(self, outputs, index):
        device = "cuda" if index.is_cuda else "cpu"
        self.label_confidence = self.label_confidence.to(device)
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.label_confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss

    def predict(self, x):
        return self.network(x)[0]

    def confidence_update(self, batchX, batchY, batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            self.label_confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
            base_value = self.label_confidence.sum(dim=1).unsqueeze(1).repeat(1, self.label_confidence.shape[1])
            self.label_confidence = self.label_confidence / base_value

class CC(Algorithm):
    """
    CC
    Reference: Provably consistent partial-label learning, NeurIPS 2020.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(CC, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        loss = self.cc_loss(self.predict(x), partial_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def cc_loss(self, outputs, partialY):
        sm_outputs = F.softmax(outputs, dim=1)
        final_outputs = sm_outputs * partialY
        average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
        return average_loss  

    def predict(self, x):
        return self.network(x)[0]



class LWS(Algorithm):
    """
    LWS
    Reference: Leveraged weighted loss for partial label learning, ICML 2021.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(LWS, self).__init__(model, input_shape, train_givenY, hparams)
        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # train_givenY = torch.from_numpy(train_givenY)
        label_confidence = torch.ones(train_givenY.shape[0], train_givenY.shape[1]) / train_givenY.shape[1]
        self.label_confidence = label_confidence

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if index.is_cuda else "cpu"
        self.label_confidence = self.label_confidence.to(device)
        loss = self.lws_loss(self.predict(x), partial_y, index)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.confidence_update(x, partial_y, index)
        return {'loss': loss.item()}

    def lws_loss(self, outputs, partialY, index):
        device = "cuda" if outputs.is_cuda else "cpu"
        onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
        onezero[partialY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)
        sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
        sig_loss1 = sig_loss1.to(device)
        sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
        sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
            1 + torch.exp(-outputs[outputs > 0]))
        l1 = self.label_confidence[index, :] * onezero * sig_loss1
        average_loss1 = torch.sum(l1) / l1.size(0)
        sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
        sig_loss2 = sig_loss2.to(device)
        sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
        sig_loss2[outputs < 0] = torch.exp(
            outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
        l2 = self.label_confidence[index, :] * counter_onezero * sig_loss2
        average_loss2 = torch.sum(l2) / l2.size(0)
        average_loss = average_loss1 + 2 * average_loss2
        return average_loss

    def predict(self, x):
        return self.network(x)[0]

    def confidence_update(self, batchX, batchY, batch_index):
        with torch.no_grad():
            device = "cuda" if batch_index.is_cuda else "cpu"
            batch_outputs = self.predict(batchX)
            sm_outputs = F.softmax(batch_outputs, dim=1)
            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[batchY > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.to(device)
            counter_onezero = counter_onezero.to(device)
            new_weight1 = sm_outputs * onezero
            new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
                self.label_confidence.shape[1], 1).transpose(0, 1)
            new_weight2 = sm_outputs * counter_onezero
            new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
                self.label_confidence.shape[1], 1).transpose(0, 1)
            new_weight = new_weight1 + new_weight2
            self.label_confidence[batch_index, :] = new_weight

class CAVL(Algorithm):
    """
    CAVL
    Reference: Exploiting Class Activation Value for Partial-Label Learning, ICLR 2022.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(CAVL, self).__init__(model, input_shape, train_givenY, hparams)
        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float()/tempY
        self.label_confidence = label_confidence
        self.label_confidence = self.label_confidence.double()

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        loss = self.rc_loss(self.predict(x), index)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.confidence_update(x, partial_y, index)
        return {'loss': loss.item()}

    def rc_loss(self, outputs, index):
        device = "cuda" if index.is_cuda else "cpu"
        self.label_confidence = self.label_confidence.to(device)
        logsm_outputs = F.log_softmax(outputs, dim=1)
        #print(self.label_confidence.is_cuda)
        final_outputs = logsm_outputs * self.label_confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss

    def predict(self, x):
        return self.network(x)[0]

    def confidence_update(self, batchX, batchY, batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            cav = (batch_outputs*torch.abs(1-batch_outputs))*batchY
            cav_pred = torch.max(cav,dim=1)[1]
            gt_label = F.one_hot(cav_pred,batchY.shape[1])
            self.label_confidence[batch_index,:] = gt_label.double()
            #self.label_confidence[batch_index,:] = gt_label.float()

class POP(Algorithm):
    """
    POP
    Reference: Progressive purification for instance-dependent partial label learning, ICML 2023.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(POP, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.train_givenY = train_givenY
        tempY = self.train_givenY.sum(dim=1).unsqueeze(1).repeat(1, self.train_givenY.shape[1])
        label_confidence = self.train_givenY.float()/tempY
        self.label_confidence = label_confidence
        self.f_record = torch.zeros([self.hparams['rollWindow'], label_confidence.shape[0], label_confidence.shape[1]])
        self.curr_iter = 0
        self.theta = self.hparams['theta']
        self.steps_per_epoch = train_givenY.shape[0] // self.hparams['batch_size']


    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if index.is_cuda else "cpu"
        loss = self.rc_loss(self.predict(x), index)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.confidence_update(x, partial_y, index)
        self.f_record = self.f_record.to(device)
        if self.curr_iter % self.steps_per_epoch == 0:
            epoch_num = self.curr_iter / self.steps_per_epoch
            self.f_record[int(epoch_num % self.hparams['rollWindow']), :] = self.label_confidence
            if self.curr_iter >= (self.hparams['warm_up'] * self.steps_per_epoch):
                temp_prob_matrix = self.f_record.mean(0)
                # label correction
                temp_prob_matrix = temp_prob_matrix / temp_prob_matrix.sum(dim=1).repeat(temp_prob_matrix.size(1),1).transpose(0, 1)
                correction_label_matrix = self.train_givenY
                correction_label_matrix = correction_label_matrix.to(device)
                pre_correction_label_matrix = correction_label_matrix.clone()
                correction_label_matrix[temp_prob_matrix / torch.max(temp_prob_matrix, dim=1, keepdim=True)[0] < self.theta] = 0
                tmp_label_matrix = temp_prob_matrix * correction_label_matrix
                self.label_confidence = tmp_label_matrix / tmp_label_matrix.sum(dim=1).repeat(tmp_label_matrix.size(1), 1).transpose(0, 1)
                if self.theta < 0.4:
                    if torch.sum(
                            torch.not_equal(pre_correction_label_matrix, correction_label_matrix)) < 0.0001 * pre_correction_label_matrix.shape[0] * self.num_classes:
                        self.theta *= (self.hparams['inc'] + 1)            
        self.curr_iter = self.curr_iter + 1

        return {'loss': loss.item()}

    def rc_loss(self, outputs, index):
        device = "cuda" if index.is_cuda else "cpu"
        self.label_confidence = self.label_confidence.to(device)
        logsm_outputs = F.log_softmax(outputs, dim=1)
        #print(self.label_confidence.is_cuda)
        final_outputs = logsm_outputs * self.label_confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss

    def predict(self, x):
        return self.network(x)[0]

    def confidence_update(self, batchX, batchY, batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            self.label_confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
            base_value = self.label_confidence.sum(dim=1).unsqueeze(1).repeat(1, self.label_confidence.shape[1])
            self.label_confidence = self.label_confidence / base_value

class IDGP(Algorithm):
    """
    IDGP
    Reference: Decompositional Generation Process for Instance-Dependent Partial Label Learning, ICLR 2023.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(IDGP, self).__init__(model, input_shape, train_givenY, hparams)
        self.model = model
        # self.featurizer_f = self.model.image_encoder
        # self.classifier_f = torch.nn.Linear(
        #     self.featurizer_f.out_dim,
        #     self.num_classes)
        # self.f = nn.Sequential(self.featurizer_f, self.classifier_f)
        self.f = self.model
        self.f_opt = torch.optim.Adam(
            self.f.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # self.featurizer_g = copy.deepcopy(self.model.image_encoder)
        # self.classifier_g = torch.nn.Linear(
        #     self.featurizer_g.out_dim,
        #     self.num_classes)
        # self.g = nn.Sequential(self.featurizer_g, self.classifier_g)
        self.g = copy.deepcopy(self.model)
        self.g_opt = torch.optim.Adam(
            self.g.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float()/tempY
        self.d_array = label_confidence
        self.b_array = train_givenY
        self.d_array = self.d_array.double()
        self.b_array = self.b_array.double()
        self.curr_iter = 0
        self.warm_up_epoch = hparams['warm_up_epoch']
        self.ramp_iter_num = int(hparams['num_epochs'] * 0.2)
        self.steps_per_epoch = train_givenY.shape[0] / self.hparams['batch_size']


    def weighted_crossentropy_f(self, f_outputs, weight, eps=1e-12):
        l = weight * torch.log(f_outputs+eps)
        loss = (-torch.sum(l)) / l.size(0)
        
        return loss

    def weighted_crossentropy_f_with_g(self, f_outputs, g_outputs, targets, eps=1e-12):
        weight = g_outputs.clone().detach() * targets
        weight[weight == 0] = 1.0
        logits1 = (1 - weight) / (weight+eps)
        logits2 = weight.prod(dim=1, keepdim=True)
        weight = logits1 * logits2
        weight = weight * targets
        weight = weight / (weight.sum(dim=1, keepdim=True)+eps)
        weight = weight.clone().detach()
        
        l = weight * torch.log(f_outputs+eps)
        loss = (-torch.sum(l)) / l.size(0)
        
        return loss

    def weighted_crossentropy_g_with_f(self, g_outputs, f_outputs, targets, eps=1e-12):
     
        weight = f_outputs.clone().detach() * targets
        weight = weight / (weight.sum(dim=1, keepdim=True) + eps)
        l = weight * ( torch.log((1 - g_outputs) / (g_outputs + eps)+eps))
        l = weight * (torch.log(1.0000001 - g_outputs))
        loss = ( - torch.sum(l)) / ( l.size(0)) + \
            ( - torch.sum(targets * torch.log(g_outputs+eps) + (1 - targets) * torch.log(1.0000001 - g_outputs))) / (l.size(0))
        
        return loss

    def weighted_crossentropy_g(self, g_outputs, weight, eps=1e-12):
        l = weight * torch.log(g_outputs+eps) + (1 - weight) * torch.log(1.0000001 - g_outputs)
        loss = ( - torch.sum(l)) / (l.size(0))

        return loss

    def update_d(self, f_outputs, targets, eps=1e-12):
        new_d = f_outputs.clone().detach() * targets.clone().detach()
        new_d = new_d / (new_d.sum(dim=1, keepdim=True) + eps)
        new_d = new_d.double()
        return new_d

    def update_b(self, g_outputs, targets):
        new_b = g_outputs.clone().detach() * targets.clone().detach()
        new_b = new_b.double()
        return new_b

    def noisy_output(self, outputs, d_array, targets):
        _, true_labels = torch.max(d_array * targets, dim=1)
        device = "cuda" if outputs.is_cuda else "cpu"
        pseudo_matrix  = F.one_hot(true_labels, outputs.shape[1]).float().to(device).detach()
        return pseudo_matrix * (1 - outputs) + (1 - pseudo_matrix) * outputs

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if index.is_cuda else "cpu"
        consistency_criterion_f = nn.KLDivLoss(reduction='batchmean').to(device)
        consistency_criterion_g = nn.KLDivLoss(reduction='batchmean').to(device)
        self.d_array = self.d_array.to(device)
        self.b_array = self.b_array.to(device)
        self.f = self.f.to(device)
        self.g = self.g.to(device)
        L_F = None
        if self.curr_iter <= self.warm_up_epoch * self.steps_per_epoch:
            # warm up of f
            f_logits_o = self.f(x)[0]
            #f_logits_o_max = torch.max(f_logits_o, dim=1)
            #f_logits_o = f_logits_o - f_logits_o_max.view(-1, 1).expand_as(f_logits_o)
            f_outputs_o = F.softmax(f_logits_o / 1., dim=1)
            L_f_o = self.weighted_crossentropy_f(f_outputs_o, self.d_array[index,:])
            L_F = L_f_o 
            self.f_opt.zero_grad()
            L_F.backward()
            self.f_opt.step()
            # warm up of g
            g_logits_o = self.g(x)[0]
            g_outputs_o = torch.sigmoid(g_logits_o / 1)
            L_g_o = self.weighted_crossentropy_g(g_outputs_o, self.b_array[index,:])
            L_g = L_g_o 
            self.g_opt.zero_grad()
            L_g.backward()
            self.g_opt.step()
        else:
            f_logits_o = self.f(x)[0]
            g_logits_o = self.g(x)[0]

            f_outputs_o = F.softmax(f_logits_o / 1., dim=1)
            g_outputs_o = torch.sigmoid(g_logits_o / 1.)

            L_f = self.weighted_crossentropy_f(f_outputs_o, self.d_array[index,:])
            L_f_g = self.weighted_crossentropy_f_with_g(f_outputs_o, self.noisy_output(g_outputs_o, self.d_array[index, :], partial_y), partial_y)
            
            L_g = self.weighted_crossentropy_g(g_outputs_o, self.b_array[index,:])
     
            L_g_f = self.weighted_crossentropy_g_with_f(g_outputs_o, f_outputs_o, partial_y)
                                            
            f_outputs_log_o = torch.log_softmax(f_logits_o, dim=-1)
            f_consist_loss0 = consistency_criterion_f(f_outputs_log_o, self.d_array[index,:].float())
            f_consist_loss = f_consist_loss0 
            g_outputs_log_o = nn.LogSigmoid()(g_logits_o)
            g_consist_loss0 = consistency_criterion_g(g_outputs_log_o, self.b_array[index,:].float())
            g_consist_loss = g_consist_loss0 
            lam = min(self.curr_iter / self.ramp_iter_num, 1)

            L_F = L_f + L_f_g + lam * f_consist_loss
            L_G = L_g + L_g_f + lam * g_consist_loss
            self.f_opt.zero_grad()
            L_F.backward()
            self.f_opt.step()
            self.g_opt.zero_grad()
            L_G.backward()
            self.g_opt.step()
        self.d_array[index,:] = self.update_d(f_outputs_o, partial_y)
        self.b_array[index,:] = self.update_b(g_outputs_o, partial_y)
        self.curr_iter += 1    

        return {'loss': L_F.item()}        

    def predict(self, x):
        return self.f(x)[0]

class ABS_MAE(Algorithm):
    """
    ABS_MAE
    Reference: On the Robustness of Average Losses for Partial-Label Learning, TPAMI 2024.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(ABS_MAE, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float()/tempY
        self.label_confidence = label_confidence

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if partial_y.is_cuda else "cpu"
        loss = self.mae_loss(self.predict(x), index, device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def mae_loss(self, outputs, index, device):
        sm_outputs = F.softmax(outputs, dim=1)
        sm_outputs = sm_outputs.unsqueeze(1)
        sm_outputs = sm_outputs.expand([-1,self.num_classes,-1])
        label_one_hot = torch.eye(self.num_classes).to(device)
        loss = torch.abs(sm_outputs - label_one_hot).sum(dim=-1)
        self.label_confidence = self.label_confidence.to(device)
        loss = loss * self.label_confidence[index, :]
        avg_loss = loss.sum(dim=1).mean()
        return avg_loss

    def predict(self, x):
        return self.network(x)[0]

class ABS_GCE(Algorithm):
    """
    ABS_GCE
    Reference: On the Robustness of Average Losses for Partial-Label Learning, TPAMI 2024.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(ABS_GCE, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float()/tempY
        self.label_confidence = label_confidence
        self.q = hparams['q']

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if partial_y.is_cuda else "cpu"
        loss = self.gce_loss(self.predict(x), index, device, q=self.q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def gce_loss(self, outputs, index, device, q):
        sm_outputs = F.softmax(outputs, dim=1)
        sm_outputs = torch.pow(sm_outputs, q)
        loss = (1. - sm_outputs) / q
        self.label_confidence = self.label_confidence.to(device)
        loss = loss * self.label_confidence[index, :]
        avg_loss = loss.sum(dim=1).mean()
        return avg_loss

    def predict(self, x):
        return self.network(x)[0]




class CRDPLL(Algorithm):
    """
    CRDPLL
    Reference: Revisiting Consistency Regularization for Deep Partial Label Learning, ICML 2022.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(CRDPLL, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence

        self.consistency_criterion = nn.KLDivLoss(reduction='batchmean')
        self.train_givenY=train_givenY
        self.lam = 1
        self.curr_iter = 0
        self.max_steps = self.hparams['num_epochs']

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        loss = self.cr_loss(self.predict(x), self.predict(strong_x), index)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.curr_iter = self.curr_iter + 1
        self.confidence_update(x,strong_x, partial_y, index)
        return {'loss': loss.item()}

    def cr_loss(self, outputs, strong_outputs, index):
        device = "cuda" if index.is_cuda else "cpu"
        self.label_confidence = self.label_confidence.to(device)
        self.consistency_criterion=self.consistency_criterion.to(device)
        self.train_givenY=self.train_givenY.to(device)
        consist_loss0 = self.consistency_criterion(F.log_softmax(outputs, dim=1), self.label_confidence[index, :].float())
        consist_loss1 = self.consistency_criterion(F.log_softmax(strong_outputs, dim=1), self.label_confidence[index, :].float())
        super_loss = -torch.mean(
            torch.sum(torch.log(1.0000001 - F.softmax(outputs, dim=1)) * (1 - self.train_givenY[index, :]), dim=1))
        lam = min((self.curr_iter / (self.max_steps*0.5)) * self.lam, self.lam)
        average_loss = lam * (consist_loss0 + consist_loss1) + super_loss
        return average_loss

    def predict(self, x):
        return self.network(x)[0]

    def confidence_update(self,batchX,strong_batchX,batchY,batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            strong_batch_outputs=self.predict(strong_batchX)
            temp_un_conf=F.softmax(batch_outputs,dim=1)
            strong_temp_un_conf=F.softmax(strong_batch_outputs,dim=1)
            self.label_confidence[batch_index,:]=torch.pow(temp_un_conf,1/(1+1))*torch.pow(strong_temp_un_conf,1/(1+1))*batchY
            base_value=self.label_confidence[batch_index,:].sum(dim=1).unsqueeze(1).repeat(1,self.label_confidence[batch_index,:].shape[1])
            self.label_confidence[batch_index,:]=self.label_confidence[batch_index,:]/base_value




class ABLE(Algorithm):
    """
    ABLE
    Reference: Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning, IJCAI 2022
    """

    class ABLE_model(nn.Module):
        def __init__(self, num_classes,input_shape,hparams, model):
            super().__init__()
            self.model = model
        def forward(self, hparams=None, img_w=None, images=None, partial_Y=None, is_eval=False):
            if is_eval:
                # output_raw, q = self.model(img_w, self.model.tuner, self.model.head)
                output_raw, q = self.model(img_w)
                return output_raw
            # outputs, features = self.model(images, self.model.tuner, self.model.head)
            outputs, features = self.model(images)
            batch_size = hparams['batch_size']
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            return outputs, features


    class ClsLoss(nn.Module):
        def __init__(self, predicted_score):
            super().__init__()
            self.predicted_score = predicted_score
            self.init_predicted_score = predicted_score.detach()
        def forward(self, outputs, index):
            device = "cuda" if outputs.is_cuda else "cpu"
            self.predicted_score=self.predicted_score.to(device)
            logsm_outputs = F.log_softmax(outputs, dim=1)
            final_outputs = self.predicted_score[index, :] * logsm_outputs
            cls_loss = - ((final_outputs).sum(dim=1)).mean()
            return cls_loss
        def update_target(self, batch_index, updated_confidence):
            with torch.no_grad():
                self.predicted_score[batch_index, :] = updated_confidence.detach()
            return None

    class ConLoss(nn.Module):
        def __init__(self, predicted_score, base_temperature=0.07):
            super().__init__()
            self.predicted_score = predicted_score
            self.init_predicted_score = predicted_score.detach()
            self.base_temperature = base_temperature
        def forward(self, hparams, outputs, features, Y, index):
            batch_size = hparams['batch_size']
            device = "cuda" if outputs.is_cuda else "cpu"
            self.predicted_score=self.predicted_score.to(device)
            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), hparams['temperature'])
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            Y = Y.float()
            output_sm = F.softmax(outputs[0: batch_size, :], dim=1).float()
            output_sm_d = output_sm.detach()
            _, target_predict = (output_sm_d * Y).max(1)
            predict_labels = target_predict.repeat(batch_size, 1).to(device)
            mask_logits = torch.zeros_like(predict_labels).float().to(device)
            pos_set = (Y == 1.0).nonzero().to(device)
            ones_flag = torch.ones(batch_size).float().to(device)
            zeros_flag = torch.zeros(batch_size).float().to(device)
            for pos_set_i in range(pos_set.shape[0]):
                sample_idx = pos_set[pos_set_i][0]
                class_idx = pos_set[pos_set_i][1]
                mask_logits_tmp = torch.where(predict_labels[sample_idx] == class_idx, ones_flag, zeros_flag).float()
                if mask_logits_tmp.sum() > 0:
                    mask_logits_tmp = mask_logits_tmp / mask_logits_tmp.sum()
                    mask_logits[sample_idx] = mask_logits[sample_idx] + mask_logits_tmp * \
                                              self.predicted_score[sample_idx][class_idx]
            mask_logits = mask_logits.repeat(anchor_count, contrast_count)
            logits_mask = torch.scatter(torch.ones_like(mask_logits),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0).float()
            mask_logits = mask_logits * logits_mask
            exp_logits = logits_mask * torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask_logits * log_prob).sum(1)
            loss_con_m = - (hparams['temperature'] / self.base_temperature) * mean_log_prob_pos
            loss_con = loss_con_m.view(anchor_count, batch_size).mean()
            revisedY_raw = Y.clone()
            revisedY_raw = revisedY_raw * output_sm_d
            revisedY_raw = revisedY_raw / revisedY_raw.sum(dim=1).repeat(Y.shape[1], 1).transpose(0, 1)
            new_target = revisedY_raw.detach()
            return loss_con, new_target
        def update_target(self, batch_index, updated_confidence):
            with torch.no_grad():
                self.predicted_score[batch_index, :] = updated_confidence.detach()
            return None

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(ABLE, self).__init__(model, input_shape, train_givenY, hparams)
        self.model = model
        self.network=self.ABLE_model(self.num_classes, input_shape, hparams, self.model)
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence
        self.loss_cls = self.ClsLoss(predicted_score=label_confidence.float())
        self.loss_con = self.ConLoss(predicted_score=label_confidence.float())
        self.train_givenY = train_givenY

    def update(self,minibatches):
        x, strong_x, partial_y, _, index = minibatches
        X_tot = torch.cat([x, strong_x], dim=0)
        batch_size = x.shape[0]

        cls_out, features = self.network(hparams=self.hparams, images=X_tot, partial_Y=partial_y, is_eval=False)
        cls_out_w = cls_out[0: batch_size, :]

        cls_loss = self.loss_cls(cls_out_w, index)
        con_loss, new_target = self.loss_con(self.hparams, cls_out, features, partial_y, index)
        loss = cls_loss + self.hparams['loss_weight'] * con_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_cls.update_target(batch_index=index, updated_confidence=new_target)
        self.loss_con.update_target(batch_index=index, updated_confidence=new_target)
        return {'loss': loss.item()}

    def predict(self,images,):
        return self.network(img_w=images, is_eval=True)

# LT-PLL
    
class Solar(Algorithm):
    """
    Solar
    Reference: SoLar: Sinkhorn Label Reffnery for Imbalanced Partial-Label Learning, ICLR 2022.
    """
    class partial_loss(nn.Module):
        def __init__(self, train_givenY):
            super().__init__()
            print('Calculating uniform targets...')
            tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
            confidence = train_givenY.float()/tempY
            confidence = confidence.cuda()
            self.confidence = confidence

        def forward(self, outputs, index, targets=None):
            logsm_outputs = F.log_softmax(outputs, dim=1)
            if targets is None:
                # using confidence
                final_outputs = logsm_outputs * self.confidence[index, :].detach()
            else:
                # using given tagets
                final_outputs = logsm_outputs * targets.detach()
            loss_vec = - ((final_outputs).sum(dim=1))
            average_loss = loss_vec.mean()
            return average_loss, loss_vec

        @torch.no_grad()
        def confidence_update(self, temp_un_conf, batch_index):
            self.confidence[batch_index, :] = temp_un_conf
            return None
    
    def __init__(self, model, input_shape, train_givenY, hparams):
        super(Solar, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float()/tempY
        self.num_class = tempY.shape[1]
        self.confidence = label_confidence.to(train_givenY.device)
        self.mu = 0.1
        self.queue = torch.zeros(hparams['queue_length'], self.num_class).cuda()
        self.emp_dist = torch.ones(self.num_class)/self.num_class
        self.emp_dist = self.emp_dist.unsqueeze(dim=1)
        self.loss_fn = self.partial_loss(train_givenY)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, epoch):
        x, strong_x, partial_y, _, index = minibatches
        
        # with autocast():
        logits_w = self.network(x)[0]
        logits_s = self.network(strong_x)[0]
        bs = x.shape[0]
        
        eta = self.hparams['eta'] * self.linear_rampup(epoch, self.hparams['warmup_epoch'])
        rho = self.hparams['rho_start'] + (self.hparams['rho_end'] - self.hparams['rho_start']) * self.linear_rampup(epoch, self.hparams['warmup_epoch'])
        # calculate weighting parameters

        prediction = F.softmax(logits_w.detach(), dim=1)
        sinkhorn_cost = prediction * partial_y
        # calculate sinkhorn cost (M matrix in our paper)
        conf_rn = sinkhorn_cost / sinkhorn_cost.sum(dim=1).repeat(prediction.size(1), 1).transpose(0, 1)
        # re-normalized prediction for unreliable examples

        # time to use queue, output now represent queue+output
        prediction_queue = sinkhorn_cost.detach()
        if self.queue is not None:
            if not torch.all(self.queue[-1, :] == 0):
                prediction_queue = torch.cat((self.queue, prediction_queue))
            # fill the queue
            self.queue[bs:] = self.queue[:-bs].clone().detach()
            self.queue[:bs] = prediction_queue[-bs:].clone().detach()
        pseudo_label_soft, flag = self.sinkhorn(prediction_queue, self.hparams['lamd'], r_in=self.emp_dist)
        pseudo_label = pseudo_label_soft[-bs:]
        pseudo_label_idx = pseudo_label.max(dim=1)[1]

        _, rn_loss_vec = self.loss_fn(logits_w, index)
        _, pseudo_loss_vec = self.loss_fn(logits_w, None, targets=pseudo_label)

        idx_chosen_sm = []
        sel_flags = torch.zeros(x.shape[0]).cuda().detach()
        # initialize selection flags
        for j in range(self.num_class):
            indices = np.where(pseudo_label_idx.cpu().numpy()==j)[0]
            # torch.where will cause device error
            if len(indices) == 0:
                continue
                # if no sample is assigned this label (by argmax), skip
            bs_j = bs * self.emp_dist[j]
            pseudo_loss_vec_j = pseudo_loss_vec[indices]
            sorted_idx_j = pseudo_loss_vec_j.sort()[1].cpu().numpy()
            partition_j = max(min(int(math.ceil(bs_j*rho)), len(indices)), 1)
            # at least one example
            idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])

        idx_chosen_sm = np.concatenate(idx_chosen_sm)
        sel_flags[idx_chosen_sm] = 1
        # filtering clean sinkhorn labels
        high_conf_cond = (pseudo_label * prediction).sum(dim=1) > self.hparams['tau']
        sel_flags[high_conf_cond] = 1
        idx_chosen = torch.where(sel_flags == 1)[0]
        idx_unchosen = torch.where(sel_flags == 0)[0]

        if epoch < 1 or idx_chosen.shape[0] == 0:
            # first epoch, using uniform labels for training
            # else, if no samples are chosen, run rn 
            loss = rn_loss_vec.mean()
        else:
            if idx_unchosen.shape[0] > 0:
                loss_unreliable = rn_loss_vec[idx_unchosen].mean()
            else:
                loss_unreliable = 0
            loss_sin = pseudo_loss_vec[idx_chosen].mean()
            loss_cons, _ = self.loss_fn(logits_s[idx_chosen], None, targets=pseudo_label[idx_chosen])
            # consistency regularization
            
            l = np.random.beta(4, 4)
            l = max(l, 1-l)
            X_w_c = x[idx_chosen]
            pseudo_label_c = pseudo_label[idx_chosen]
            idx = torch.randperm(X_w_c.size(0))
            X_w_c_rand = X_w_c[idx]
            pseudo_label_c_rand = pseudo_label_c[idx]
            X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand        
            pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
            logits_mix, _ = self.network(X_w_c_mix)
            loss_mix, _  = self.loss_fn(logits_mix, None, targets=pseudo_label_c_mix)
            # mixup training

            loss = (loss_sin + loss_mix + loss_cons) * eta + loss_unreliable * (1 - eta)
            
            # self.emp_dist = self.confidence.sum(0) / self.confidence.sum()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.confidence_move_update(conf_rn, index)
        
        return loss

    @torch.no_grad()
    def confidence_move_update(self, temp_un_conf, batch_index, ratio=None):
        self.confidence = self.confidence.to(batch_index.device)
        if ratio:
            self.confidence[batch_index, :] = self.confidence[batch_index, :] * (1 - ratio) + temp_un_conf * ratio
        else:
            self.confidence[batch_index, :] = self.confidence[batch_index, :] * (1 - self.mu) + temp_un_conf * self.mu
        return None

    def distribution_update(self, est_loader):
        gamma = 0.01
        with torch.no_grad():
            print('==> Estimating empirical label distribution ...')       
            self.network.eval()
            est_pred_list = []
            for _, (images, _, labels, true_labels, index) in enumerate(est_loader):
                images, labels = images.cuda(), labels.cuda()
                outputs, _ = self.network(images)   
                pred = torch.softmax(outputs, dim=1) * labels
                est_pred_list.append(pred.cpu())
            
        est_pred_idx = torch.cat(est_pred_list, dim=0).max(dim=1)[1]
        est_pred = F.one_hot(est_pred_idx, self.num_class).detach()
        emp_dist = est_pred.sum(0)
        emp_dist = emp_dist / float(emp_dist.sum())

        emp_dist_train = emp_dist.unsqueeze(1)
        # estimating empirical class prior by counting prediction
        self.emp_dist = emp_dist_train * gamma + self.emp_dist * (1 - gamma)
        # moving-average updating class prior
        
        
        
    def predict(self, x):
        return self.network(x)[0]
    
    def sinkhorn(self, pred, eta, r_in=None, rec=False):
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("pred contains NaN!")
            
        if r_in is not None:
            if torch.isnan(r_in).any() or torch.isinf(r_in).any():
                print("r_in contains NaN!")
                            
        PS = pred.detach()
        K = PS.shape[1]
        N = PS.shape[0]
        PS = PS.T
        c = torch.ones((N, 1)) / N
        r = r_in.cuda()
        c = c.cuda()
        # average column mean 1/N
        PS = torch.pow(PS, eta)  # K x N
        r_init = copy.deepcopy(r)
        inv_N = 1. / N
        err = 1e6
        # error rate
        _counter = 1
        for i in range(50):
            if err < 1e-1:
                break
            r = r_init * (1 / (PS @ c))  # (KxN)@(N,1) = K x 1
            # 1/K(Plambda * beta)
            c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
            # 1/N(alpha * Plambda)
            if _counter % 10 == 0:
                err = torch.sum(c_new) + torch.sum(r)
                if torch.isnan(err):
                    # This may very rarely occur (maybe 1 in 1k epochs)
                    # So we do not terminate it, but return a relaxed solution
                    print('====> Nan detected, return relaxed solution')
                    pred_new = pred + 1e-5 * (pred == 0)
                    relaxed_PS, _ = self.sinkhorn(pred_new, eta, r_in=r_in, rec=True)
                    z = (1.0 * (pred != 0))
                    relaxed_PS = relaxed_PS * z
                    return relaxed_PS, True
            c = c_new
            _counter += 1
        PS *= torch.squeeze(c)
        PS = PS.T
        PS *= torch.squeeze(r)
        PS *= N
        return PS.detach(), False
    
    def linear_rampup(self, current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length
        
        

class HTC(Algorithm):
    """
    HTC
    Reference: Long-tailed Partial Label Learning by Head Classiffer and Tail Classiffer Cooperation, AAAI 2024.
    """
    class PLL_loss(nn.Module):
        def __init__(self, train_givenY, mu=0.1):
            super().__init__()
            self.mu = mu
            print('Calculating uniform targets...')
            # calculate confidence
            self.device = train_givenY.device
            self.confidence = train_givenY.float()/train_givenY.sum(dim=1, keepdim=True).to(self.device)
            self.distribution = self.confidence.sum(0)/self.confidence.sum().to(self.device)

        def forward(self, logits, index, targets=None):
            log_p = F.log_softmax(logits, dim=1)
            if targets is None:
                # using confidence
                self.confidence = self.confidence.to(log_p.device)
                final_outputs = log_p * self.confidence[index, :].detach()
            else:
                # using given tagets
                final_outputs = log_p * targets.detach()
            loss_vec = -final_outputs.sum(dim=1)
            average_loss = loss_vec.mean()
            return average_loss, loss_vec

        @torch.no_grad()
        def get_distribution(self):
            self.update_distribution()
            return self.distribution

        @torch.no_grad()
        def update_distribution(self):
            self.distribution = self.confidence.sum(0) / self.confidence.sum()
    
    
    class DHNet_Atten(nn.Module):
        def __init__(self, num_class, encoder):
            super().__init__()

            self.encoder = encoder
            self.feat_dim = self.encoder.out_dim
            
            self.fc_head = nn.Linear(self.feat_dim, num_class)
            self.fc_tail = nn.Linear(self.feat_dim, num_class)
            self.attention = nn.Linear(num_class * 2, 2)

        def forward(self, x):
            feat = self.encoder(x)
            logit_tail = self.fc_tail(feat)
            logit_head = self.fc_head(feat)
            return logit_head, logit_tail, feat

        def ensemble(self, logit_head, logit_tail, distribution):
            p1, p2 = F.softmax(logit_head - torch.log(distribution+1e-5), dim=1), F.softmax(logit_tail, dim=1)
            weights = F.softmax(F.leaky_relu(self.attention(torch.cat([p1, p2], dim=1))), dim=1)
            w1, w2 = torch.split(F.normalize(weights, p=2), 1, dim=1)
            pred = F.softmax(w1 * p1 + w2 * p2, dim=1)
            return pred
    
    
    def __init__(self, model, input_shape, train_givenY, hparams):
        super(HTC, self).__init__(model, input_shape, train_givenY, hparams)

        # self.model = model
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float()/tempY
        self.confidence = label_confidence.to(train_givenY.device)
        self.num_class = tempY.shape[1]
        # self.network = self.DHNet_Atten(self.num_class, self.model.image_encoder)
        self.network = model
        self.mu = 0.1
        self.emp_dist_head = label_confidence
        self.emp_dist_tail = torch.Tensor([1 / self.num_class for _ in range(self.num_class)])
        self.loss_fn = self.PLL_loss(train_givenY).to(train_givenY.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, epoch):
        x, strong_x, Y, _, index = minibatches
        
        with autocast():
            logits_w_head, logits_w_tail, feat_w = self.network(x)
            logits_s_head, logits_s_tail, feat_s = self.network(strong_x)
            self.confidence = self.confidence.to(index.device)
            pseudo_label = self.confidence[index]

            self.eta = 0.9 * self.linear_rampup(epoch, self.hparams["num_epochs"])
            self.alpha = 0.2 + (0.6 - 0.2) * self.linear_rampup(epoch, self.hparams["num_epochs"])
            # self.emp_dist_head = self.confidence.sum(0) / self.confidence.sum()
            loss_head, prediction_head = self.get_loss(x, logits_w_head, logits_s_head, pseudo_label, Y, index, self.network,
                                                        self.loss_fn, self.emp_dist_head, self.alpha, self.eta, epoch, False)

            logit_adj = F.softmax(logits_w_head - 2 * torch.log(self.emp_dist_head), dim=1)
            loss_tail, prediction_tail = self.get_loss(x, logits_w_tail, logits_s_tail, logit_adj, Y, index, self.network,
                                                        self.loss_fn, self.emp_dist_tail, self.alpha, self.eta, epoch, True)

            fusion_pred = self.network.ensemble(logits_w_head.detach(), logits_w_tail.detach(), self.emp_dist_head)
            fusion_loss = torch.sum(-pseudo_label * torch.log(fusion_pred+1e-8))/fusion_pred.shape[0]
            ratio = 0.5 * self.linear_rampup(epoch, self.hparams["num_epochs"])
            loss = loss_head + loss_tail + ratio * fusion_loss
            self.confidence_move_update(prediction_tail, index)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss
    
    @torch.no_grad()
    def confidence_move_update(self, temp_un_conf, batch_index, ratio=None):
        if ratio:
            self.confidence[batch_index, :] = self.confidence[batch_index, :] * (1 - ratio) + temp_un_conf * ratio
        else:
            self.confidence[batch_index, :] = self.confidence[batch_index, :] * (1 - self.mu) + temp_un_conf * self.mu
        return None
    
    def get_high_confidence(self, loss_vec,  pseudo_label_idx, nums_vec):
        idx_chosen = []
        chosen_flags = torch.zeros(len(loss_vec)).cuda()
        # initialize selection flags
        for j, nums in enumerate(nums_vec):
            indices = np.where(pseudo_label_idx.cpu().numpy() == j)[0]
            # torch.where will cause device error
            if len(indices) == 0:
                continue
                # if no sample is assigned this label1 (by argmax), skip
            loss_vec_j = loss_vec[indices]
            sorted_idx_j = loss_vec_j.sort()[1].cpu().numpy()
            partition_j = max(min(int(math.ceil(nums)), len(indices)), 1)
            # at least one example
            idx_chosen.append(indices[sorted_idx_j[:partition_j]])

        idx_chosen = np.concatenate(idx_chosen)
        chosen_flags[idx_chosen] = 1

        idx_chosen = torch.where(chosen_flags == 1)[0]
        return idx_chosen

    def get_loss(self, X_w, logits_w, logits_s, ce_label, Y, index, model, loss_fn, emp_dist, alpha, eta, epoch,
                 is_tail):
        bs = X_w.shape[0]
        prediction = F.softmax(logits_w.detach(), dim=1)
        prediction_adj = prediction * Y
        prediction_adj = prediction_adj / prediction_adj.sum(dim=1, keepdim=True)
        # re-normalized prediction for unreliable examples

        _, ce_loss_vec = loss_fn(logits_w, None, targets=ce_label)
        loss_pseu, _ = loss_fn(logits_w, index)

        pseudo_label_idx = ce_label.max(dim=1)[1]
        r_vec = emp_dist * bs * alpha
        idx_chosen = self.get_high_confidence(ce_loss_vec, pseudo_label_idx, r_vec.tolist())

        if epoch < 1 or idx_chosen.shape[0] == 0:
            # first epoch, using uniform labels for training
            #if no samples are chosen
            loss = loss_pseu
        else:
            loss_ce, _ = loss_fn(logits_s[idx_chosen], None, targets=ce_label[idx_chosen])
            # consistency regularization

            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            X_w_c = X_w[idx_chosen]
            ce_label_c = ce_label[idx_chosen]
            idx = torch.randperm(X_w_c.size(0))
            X_w_c_rand = X_w_c[idx]
            ce_label_c_rand = ce_label_c[idx]
            X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
            ce_label_c_mix = l * ce_label_c + (1 - l) * ce_label_c_rand
            if is_tail:
                _, logits_mix, _ = model(X_w_c_mix)
            else:
                logits_mix, _, _ = model(X_w_c_mix)
            loss_mix, _ = loss_fn(logits_mix, None, targets=ce_label_c_mix)
            # mixup training
            
            loss = (loss_mix + loss_ce) * eta + loss_pseu
            
        return loss, prediction_adj
    
    def linear_rampup(self, current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length
        
        
        
class RECORDS(Algorithm):
    """
    RECORDS
    Reference: Revisiting Consistency Regularization for Deep Partial Label Learning, ICML 2022.
    """
    
    class CORR_loss_RECORDS_mixup(nn.Module):
        def __init__(self, target, m = 0.9, mixup=0.5):
            super().__init__()
            self.confidence = target
            self.init_confidence = target.clone()
            self.feat_mean = None
            self.m = m
            self.mixup = mixup


        def forward(self,output_w,output_s,output_w_mix,output_s_mix,feat_w,feat_s,model,index,pseudo_label_mix,update_target=True):
            pred_s = F.softmax(output_s, dim=1)
            pred_w = F.softmax(output_w, dim=1)
            target = self.confidence[index, :]
            neg = (target==0).float()
            sup_loss = neg * (-torch.log(abs(1-pred_w)+1e-9)-torch.log(abs(1-pred_s)+1e-9))
            if torch.any(torch.isnan(sup_loss)):
                print("sup_loss:nan")
            sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
            con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
            if update_target:
                pred_s_mix = F.softmax(output_s_mix, dim=1)
                pred_w_mix = F.softmax(output_w_mix, dim=1)
                neg2 = (pseudo_label_mix==0).float()
                sup_loss2 = neg2 * (-torch.log(abs(1-pred_w_mix)+1e-9)-torch.log(abs(1-pred_s_mix)+1e-9))
                sup_loss_2 = torch.sum(sup_loss2) / sup_loss2.size(0)
                con_loss_2 = F.kl_div(torch.log_softmax(output_w_mix,dim=1),pseudo_label_mix,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s_mix,dim=1),pseudo_label_mix,reduction='batchmean')
            else:
                sup_loss_2 = 0
                con_loss_2 = 0

            if torch.any(torch.isnan(con_loss)):
                print("con_loss:nan")
            loss = sup_loss1 + con_loss + self.mixup * (sup_loss_2+ con_loss_2)

            if self.feat_mean is None:
                self.feat_mean = (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)
            else:
                self.feat_mean = self.m*self.feat_mean + (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)
            
            
            if update_target:
                bias = model.head(self.feat_mean.unsqueeze(0)).detach()
                bias = F.softmax(bias, dim=1)
                logits_s = output_s - torch.log(bias + 1e-9) 
                logits_w = output_w - torch.log(bias + 1e-9) 
                pred_s = F.softmax(logits_s, dim=1)
                pred_w = F.softmax(logits_w, dim=1)


                # revisedY = target.clone()
                revisedY = self.init_confidence[index,:].clone()
                revisedY[revisedY > 0]  = 1
                revisedY_s = revisedY * pred_s
                resisedY_w = revisedY * pred_w
                revisedY = revisedY_s * resisedY_w            
                revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

                # sqr
                revisedY = torch.sqrt(revisedY)
                revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

                new_target = revisedY

                self.confidence[index,:]=new_target.detach()

            return loss

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(RECORDS, self).__init__(model, input_shape, train_givenY, hparams)

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence

        self.consistency_criterion = nn.KLDivLoss(reduction='batchmean')
        self.train_givenY=train_givenY
        self.lam = 1
        self.curr_iter = 0
        self.max_steps = self.hparams['num_epochs']
        
        self.criterion = self.CORR_loss_RECORDS_mixup(self.label_confidence, m=0.9, mixup=1.0)

    def update(self, minibatches):
        x, X_s, partial_y, _, index = minibatches
        self.criterion = self.criterion.to(index.device)
        self.criterion.confidence = self.criterion.confidence.to(index.device)
        self.criterion.init_confidence = self.criterion.init_confidence.to(index.device)
        if self.criterion.feat_mean is not None:
            self.criterion.feat_mean = self.criterion.feat_mean.to(index.device)
        pseudo_label = self.criterion.confidence[index,:].clone().detach()
        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        idx = torch.randperm(x.size(0))
        X_w_rand = x[idx]
        X_s_rand = X_s[idx]
        pseudo_label_rand = pseudo_label[idx]
        X_w_mix = l * x + (1 - l) * X_w_rand   
        X_s_mix = l * X_s + (1 - l) * X_s_rand  
        pseudo_label_mix = l * pseudo_label + (1 - l) * pseudo_label_rand  
            
        with autocast():
            cls_out, feat = self.model(torch.cat((x, X_s, X_w_mix, X_s_mix), 0))
            batch_size = x.shape[0]
            cls_out_w, cls_out_s, cls_out_w_mix, cls_out_s_mix = torch.split(cls_out, batch_size, dim=0)
            feat_w, feat_s, _, _ = torch.split(feat, batch_size, dim=0)
            loss = self.criterion(cls_out_w, cls_out_s, cls_out_w_mix, cls_out_s_mix, feat_w, feat_s, self.model, index, pseudo_label_mix, True)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {'loss': loss.item()}


    def predict(self, x):
        return self.network(x)[0]

    def confidence_update(self,batchX,strong_batchX,batchY,batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            strong_batch_outputs=self.predict(strong_batchX)
            temp_un_conf=F.softmax(batch_outputs,dim=1)
            strong_temp_un_conf=F.softmax(strong_batch_outputs,dim=1)
            self.label_confidence[batch_index,:]=torch.pow(temp_un_conf,1/(1+1))*torch.pow(strong_temp_un_conf,1/(1+1))*batchY
            base_value=self.label_confidence[batch_index,:].sum(dim=1).unsqueeze(1).repeat(1,self.label_confidence[batch_index,:].shape[1])
            self.label_confidence[batch_index,:]=self.label_confidence[batch_index,:]/base_value
            
    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather, dim=0)
    
    return output