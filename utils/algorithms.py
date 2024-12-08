import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import random
import scipy.sparse as sp
import os
from sklearn.metrics import euclidean_distances

# from . import networks

ALGORITHMS = [
    'PRODEN',
    'VALEN',
    'CAVL',
    'POP',
    'ABS_MAE',
    'ABS_GCE',
    'CC',
    'EXP',
    'MCL_GCE',
    'MCL_MSE',
    'LWS',
    'IDGP',
    'PC',
    'Forward',
    'NN',
    'GA',
    'SCL_EXP',
    'SCL_NL',
    'L_W',
    'OP_W',    
    'PiCO',    
    'ABLE',
    'CRDPLL',
    'DIRK',
    'FREDIS',
    'ALIM',
    'PiCO_plus',
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

class EXP(Algorithm):
    """
    EXP
    Reference: Learning with multiple complementary labels, ICML 2020.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(EXP, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        loss = self.exp_loss(self.predict(x), partial_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def exp_loss(self, outputs, partialY):
        can_num = partialY.sum(dim=1).float() # n        
        soft_max = nn.Softmax(dim=1)
        sm_outputs = soft_max(outputs)
        final_outputs = sm_outputs * partialY
        average_loss = ((self.num_classes-1)/(self.num_classes-can_num) * torch.exp(-final_outputs.sum(dim=1))).mean()
        return average_loss  

    def predict(self, x):
        return self.network(x)[0]

class URE_LMCL(Algorithm):
    """
    URE for LMCL
    Reference: Learning with multiple complementary labels, ICML 2020.
    """
    def __init__(self, model, input_shape, train_givenY, hparams):
        super(URE_LMCL, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        loss = self.unbiased_estimator(self.predict(x), partial_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def unbiased_estimator(self, outputs, partialY):
        device = "cuda" if outputs.is_cuda else "cpu"
        comp_num = self.num_classes - partialY.sum(dim=1)
        temp_loss = torch.zeros_like(outputs).to(device)
        for i in range(self.num_classes):
            tempY = torch.zeros_like(outputs).to(device)
            tempY[:, i] = 1.0
            temp_loss[:, i] = self.loss_fn(outputs, tempY)

        candidate_loss = (temp_loss * partialY).sum(dim=1)
        noncandidate_loss = (temp_loss * (1-partialY)).sum(dim=1)
        total_loss = candidate_loss - (self.num_classes-comp_num-1.0)/(comp_num * noncandidate_loss+1e-20)
        average_loss = total_loss.mean()
        return average_loss

    def loss_fn(self, outputs, Y):

        raise NotImplementedError

    def predict(self, x):
        return self.network(x)[0]

class MCL_GCE(URE_LMCL):
    """
    MCL_GCE
    Reference: Learning with multiple complementary labels, ICML 2020.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(MCL_GCE, self).__init__(model, input_shape, train_givenY, hparams)

    def loss_fn(self, outputs, Y):
        q = 0.7
        sm_outputs = F.softmax(outputs, dim=1)
        pow_outputs = torch.pow(sm_outputs, q)
        sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q # n
        return sample_loss

class MCL_MSE(URE_LMCL):
    """
    MCL_MSE
    Reference: Learning with multiple complementary labels, ICML 2020.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(MCL_MSE, self).__init__(model, input_shape, train_givenY, hparams)

    def loss_fn(self, outputs, Y):
        sm_outputs = F.softmax(outputs, dim=1)
        loss_fn_local = nn.MSELoss(reduction='none')
        loss_matrix = loss_fn_local(sm_outputs, Y.float())
        sample_loss = loss_matrix.sum(dim=-1)
        return sample_loss

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
            #print(self.label_confidence.dtype)
            #print(gt_label.float().dtype)
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
        #self.train_givenY = torch.from_numpy(train_givenY)
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
        self.featurizer_f = networks.Featurizer(input_shape, self.hparams)
        self.classifier_f = networks.Classifier(
            self.featurizer_f.n_outputs,
            self.num_classes)
        self.f = nn.Sequential(self.featurizer_f, self.classifier_f)
        self.f_opt = torch.optim.Adam(
            self.f.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.featurizer_g = networks.Featurizer(input_shape, self.hparams)
        self.classifier_g = networks.Classifier(
            self.featurizer_g.n_outputs,
            self.num_classes)
        self.g = nn.Sequential(self.featurizer_g, self.classifier_g)
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
        L_F = None
        if self.curr_iter <= self.warm_up_epoch * self.steps_per_epoch:
            # warm up of f
            f_logits_o = self.f(x)
            #f_logits_o_max = torch.max(f_logits_o, dim=1)
            #f_logits_o = f_logits_o - f_logits_o_max.view(-1, 1).expand_as(f_logits_o)
            f_outputs_o = F.softmax(f_logits_o / 1., dim=1)
            L_f_o = self.weighted_crossentropy_f(f_outputs_o, self.d_array[index,:])
            L_F = L_f_o 
            self.f_opt.zero_grad()
            L_F.backward()
            self.f_opt.step()
            # warm up of g
            g_logits_o = self.g(x)
            g_outputs_o = torch.sigmoid(g_logits_o / 1)
            L_g_o = self.weighted_crossentropy_g(g_outputs_o, self.b_array[index,:])
            L_g = L_g_o 
            self.g_opt.zero_grad()
            L_g.backward()
            self.g_opt.step()
        else:
            f_logits_o = self.f(x)
            g_logits_o = self.g(x)

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
        return self.f(x)

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

class DIRK(Algorithm):
    """
    DIRK
    Reference: Distilling Reliable Knowledge for Instance-dependent Partial Label Learning, AAAI 2024
    """

    class tea_model(nn.Module):
        def __init__(self, num_classes,input_shape,hparams, base_encoder):
            super().__init__()
            self.encoder = base_encoder(num_classes,input_shape,hparams)
            self.register_buffer("queue_feat", torch.randn(hparams['moco_queue'], hparams['feat_dim']))
            self.register_buffer("queue_dist", torch.randn(hparams['moco_queue'], num_classes))
            self.register_buffer("queue_partY", torch.randn(hparams['moco_queue'], num_classes))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.queue_feat = F.normalize(self.queue_feat, dim=0)
            self.moco_queue = hparams['moco_queue']

        @torch.no_grad()
        def _dequeue_and_enqueue(self, keys_feat, keys_dist, keys_partY):
            batch_size = keys_feat.shape[0]
            ptr = int(self.queue_ptr)
            assert self.moco_queue % batch_size == 0
            self.queue_feat[ptr:ptr + batch_size] = keys_feat
            self.queue_dist[ptr:ptr + batch_size] = keys_dist
            self.queue_partY[ptr:ptr + batch_size] = keys_partY
            ptr = (ptr + batch_size) % self.moco_queue
            self.queue_ptr[0] = ptr

        def forward(self, img_w=None, img_s=None, img_distill=None, partY=None):
            with torch.no_grad():
                _, feat_k = self.encoder(img_w)
                output_k, _ = self.encoder(img_distill)
                output_k = torch.softmax(output_k, dim=1)
                output_k = self.get_correct_conf(output_k, partY)

            features = torch.cat((feat_k, self.queue_feat.clone().detach()), dim=0)
            partYs = torch.cat((partY, self.queue_partY.clone().detach()), dim=0)
            dists = torch.cat((output_k, self.queue_dist.clone().detach()), dim=0)
            self._dequeue_and_enqueue(feat_k, output_k, partY)
            return features, partYs, dists, output_k

        def get_correct_conf(self, un_conf, partY):
            part_confidence = un_conf * partY
            part_confidence = part_confidence / part_confidence.sum(dim=1).unsqueeze(1).repeat(1,part_confidence.shape[1])
            comp_confidence = un_conf * (1 - partY)
            comp_confidence = comp_confidence / (comp_confidence.sum(dim=1).unsqueeze(1).repeat(1, comp_confidence.shape[1]) + 1e-20)
            comp_max = comp_confidence.max(dim=1)[0].unsqueeze(1).repeat(1, partY.shape[1])
            part_min = ((1 - partY) + part_confidence).min(dim=1)[0].unsqueeze(1).repeat(1, partY.shape[1])
            fenmu = (un_conf * partY).sum(dim=1)
            M = 1.0 / fenmu
            M = M.unsqueeze(1).repeat(1, partY.shape[1])
            a = (M * comp_max) / (M * comp_max + part_min)
            a[a == 0] = 1
            rec_confidence = part_confidence * a + comp_confidence * (1 - a)
            return rec_confidence

    class stu_model(nn.Module):
        def __init__(self, num_classes,input_shape,hparams, base_encoder):
            super().__init__()
            self.encoder = base_encoder(num_classes,input_shape,hparams)
        def forward(self, img_s, img_distill, is_eval=False):
            output_s, _ = self.encoder(img_distill)
            if is_eval:
                return output_s
            _, feat_s = self.encoder(img_s)
            return output_s, feat_s

    class DIRKNet(nn.Module):
        def __init__(self, num_classes, input_shape, hparams):
            super().__init__()
            self.featurizer = networks.Featurizer(input_shape, hparams)
            self.classifier = networks.Classifier(
                self.featurizer.n_outputs,
                num_classes)
            self.head = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, hparams['feat_dim']))
        def forward(self, x):
            feat = self.featurizer(x)
            feat_c = self.head(feat)
            logits = self.classifier(feat)
            return logits, F.normalize(feat_c, dim=1)

    def dirk_loss(self,output, confidence, Y=None):
        logsm_outputs = F.log_softmax(output, dim=1)
        final_outputs = logsm_outputs * confidence
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss


    class WeightedConLoss(nn.Module):
        def __init__(self, temperature=0.07, base_temperature=0.07, dist_temperature=0.07):
            super().__init__()
            self.temperature = temperature
            self.base_temperature = base_temperature
            self.dist_temperature = dist_temperature
        def forward(self, features, dist, mask=None, batch_size=-1):
            if mask is not None:
                mask = mask.float()
                anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T),self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()
                logits_mask = torch.scatter(torch.ones_like(anchor_dot_contrast),1,torch.arange(batch_size).view(-1, 1).cuda(),0)
                mask = logits_mask * mask
                dist_temperature = self.dist_temperature
                dist_norm = dist / torch.norm(dist, dim=-1, keepdim=True)
                anchor_dot_simi = torch.div(torch.matmul(dist_norm[:batch_size], dist_norm.T), dist_temperature)
                logits_simi_max, _ = torch.max(anchor_dot_simi, dim=1, keepdim=True)
                logits_simi = anchor_dot_simi - logits_simi_max.detach()
                exp_simi = torch.exp(logits_simi) * mask
                weight = exp_simi / exp_simi.sum(dim=1).unsqueeze(1).repeat(1, anchor_dot_simi.shape[1])
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
                weighted_log_prob_pos = weight * log_prob
                loss = -(self.temperature / self.base_temperature) * weighted_log_prob_pos
                loss = loss.sum(dim=1).mean()
            else:
                q = features[:batch_size]
                k = features[batch_size:batch_size * 2]
                queue = features[batch_size * 2:]
                k, queue = k.detach(), queue.detach()
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                l_neg = torch.einsum('nc,kc->nk', [q, queue])
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                loss = F.cross_entropy(logits, labels)
            return loss

    def __init__(self,input_shape,train_givenY,hparams):
        super(DIRK, self).__init__(input_shape,train_givenY,hparams)
        self.stu = self.stu_model(self.num_classes,input_shape,hparams,self.DIRKNet)
        self.tea=self.tea_model(self.num_classes,input_shape,hparams,self.DIRKNet)
        self.optimizer = torch.optim.Adam(
            self.stu.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # self.train_givenY = torch.from_numpy(train_givenY)
        self.loss_cont_fn = self.WeightedConLoss(temperature=self.hparams['feat_temperature'], dist_temperature=self.hparams['dist_temperature'])
        self.curr_iter = 0


    def update(self,minibatches):
        x, strong_x, partial_y, _, index = minibatches

        features, partYs, dists, rec_conf_t = self.tea(x, strong_x, distill_x, partial_y)
        output_s, feat_s = self.stu(strong_x, distill_x)
        features_cont = torch.cat((feat_s, features), dim=0)
        partY_cont = torch.cat((partial_y, partYs), dim=0)
        dist_cont = torch.cat((rec_conf_t, dists), dim=0)
        batch_size = output_s.shape[0]
        mask_partial = torch.matmul(partY_cont[:batch_size], partY_cont.T)
        mask_partial[mask_partial != 0] = 1
        _, pseudo_target = torch.max(dist_cont, dim=1)
        pseudo_target = pseudo_target.contiguous().view(-1, 1)
        mask_pseudo_target = torch.eq(pseudo_target[:batch_size], pseudo_target.T).float()
        start_upd_prot = self.curr_iter >= self.hparams['prot_start']
        if start_upd_prot:
            mask = mask_partial * mask_pseudo_target
        else:
            mask = None

        if self.hparams['weight'] != 0:
            loss_cont = self.loss_cont_fn(features=features_cont, dist=dist_cont, mask=mask, batch_size=partial_y.shape[0])
        else:
            loss_cont = torch.tensor(0.0).cuda()
        loss_dirk = self.dirk_loss(output_s, rec_conf_t)
        loss = loss_dirk + self.hparams['weight'] * loss_cont

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.curr_iter += 1
        self.model_update(self.tea,self.stu,self.hparams['momentum'])
        return {'loss': loss.item()}

    def predict(self, x):
        return self.stu(None,x,is_eval=True)

    def tea_predict(self,x):
        return self.tea(x)

    def model_update(self,model_tea, model_stu, momentum=0.99):
        for param_tea, param_stu in zip(model_tea.parameters(), model_stu.parameters()):
            param_tea.data = param_tea.data * momentum + param_stu.data * (1 - momentum)



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
        def __init__(self, num_classes,input_shape,hparams, base_encoder):
            super().__init__()
            self.encoder = base_encoder(num_classes,input_shape,hparams)
        def forward(self, hparams=None, img_w=None, images=None, partial_Y=None, is_eval=False):
            if is_eval:
                output_raw, q = self.encoder(img_w)
                return output_raw
            outputs, features = self.encoder(images)
            batch_size = hparams['batch_size']
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            return outputs, features

    class ABLENet(nn.Module):
        def __init__(self,num_classes,input_shape,hparams):
            super().__init__()
            self.featurizer = networks.Featurizer(input_shape, hparams)
            self.classifier = networks.Classifier(
                self.featurizer.n_outputs,
                num_classes)
            self.head = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, hparams['feat_dim']))
        def forward(self,x):
            feat = self.featurizer(x)
            feat_c = self.head(feat)
            logits = self.classifier(feat)
            return logits, F.normalize(feat_c, dim=1)

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

    def __init__(self,input_shape,train_givenY,hparams):
        super(ABLE, self).__init__(input_shape,train_givenY,hparams)
        self.network=self.ABLE_model(self.num_classes,input_shape,hparams=hparams,base_encoder=self.ABLENet)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence
        self.loss_cls = self.ClsLoss(predicted_score=label_confidence.float())
        self.loss_con = self.ConLoss(predicted_score=label_confidence.float())
        self.train_givenY = train_givenY

    def update(self,minibatches):
        x, strong_x, partial_y, _, index = minibatches
        X_tot = torch.cat([x, strong_x], dim=0)
        batch_size = self.hparams['batch_size']

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
        return self.network(img_w=images,is_eval=True)


class PiCO(Algorithm):
    """
    PiCO
    Reference: PiCO: Contrastive Label Disambiguation for Partial Label Learning
    """

    class PiCO_model(nn.Module):
        def __init__(self, num_classes,input_shape,hparams, base_encoder):
            super().__init__()
            self.encoder_q = base_encoder(num_classes, input_shape, hparams)
            self.encoder_k = base_encoder(num_classes, input_shape, hparams)
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            self.register_buffer("queue", torch.randn(hparams['moco_queue'], hparams['feat_dim']))
            self.register_buffer("queue_pseudo", torch.randn(hparams['moco_queue']))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("prototypes", torch.zeros(num_classes, hparams['feat_dim']))
            self.queue = F.normalize(self.queue, dim=0)

        @torch.no_grad()
        def _momentum_update_key_encoder(self, hparams):
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * hparams['moco_m'] + param_q.data * (1. - hparams['moco_m'])

        @torch.no_grad()
        def _dequeue_and_enqueue(self, keys, labels, hparams):
            batch_size = keys.shape[0]
            ptr = int(self.queue_ptr)
            assert hparams['moco_queue'] % batch_size == 0
            self.queue[ptr:ptr + batch_size, :] = keys
            self.queue_pseudo[ptr:ptr + batch_size] = labels
            ptr = (ptr + batch_size) % hparams['moco_queue']
            self.queue_ptr[0] = ptr


        def forward(self, img_q, im_k=None, partial_Y=None, hparams=None, is_eval=False):
            output, q = self.encoder_q(img_q)
            if is_eval:
                return output

            predicted_scores = torch.softmax(output, dim=1) * partial_Y
            max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
            prototypes = self.prototypes.clone().detach()
            logits_prot = torch.mm(q, prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)
            with torch.no_grad():
                for feat, label in zip(q, pseudo_labels_b):
                    self.prototypes[label] = self.prototypes[label] * hparams['proto_m'] + (1 - hparams['proto_m']) * feat
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1).detach()
            with torch.no_grad():
                self._momentum_update_key_encoder(hparams)
                _, k = self.encoder_k(im_k)
            features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
            pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
            self._dequeue_and_enqueue(k, pseudo_labels_b, hparams)
            return output, features, pseudo_labels, score_prot


    class PiCONet(nn.Module):
        def __init__(self,num_classes,input_shape,hparams):
            super().__init__()
            self.featurizer = networks.Featurizer(input_shape, hparams)
            self.classifier = networks.Classifier(
                self.featurizer.n_outputs,
                num_classes)
            self.head = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, hparams['feat_dim']))
            self.register_buffer("prototypes", torch.zeros(num_classes, hparams['feat_dim']))

        def forward(self, x):
            feat = self.featurizer(x)
            feat_c = self.head(feat)
            logits = self.classifier(feat)
            return logits, F.normalize(feat_c, dim=1)

    class partial_loss(nn.Module):
        def __init__(self, confidence, hparams, conf_ema_m=0.99):
            super().__init__()
            self.confidence = confidence
            self.init_conf = confidence.detach()
            self.conf_ema_m = conf_ema_m
            self.conf_ema_range = [float(item) for item in hparams['conf_ema_range'].split(',')]
        def set_conf_ema_m(self, epoch, total_epochs):
            start = self.conf_ema_range[0]
            end = self.conf_ema_range[1]
            self.conf_ema_m = 1. * epoch /total_epochs * (end - start) + start
        def forward(self, outputs, index):
            device = "cuda" if outputs.is_cuda else "cpu"
            self.confidence=self.confidence.to(device)
            logsm_outputs = F.log_softmax(outputs, dim=1)
            final_outputs = logsm_outputs * self.confidence[index, :]
            average_loss = - ((final_outputs).sum(dim=1)).mean()
            return average_loss
        def confidence_update(self, temp_un_conf, batch_index, batchY):
            with torch.no_grad():
                _, prot_pred = (temp_un_conf * batchY).max(dim=1)
                pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
                self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] \
                                                  + (1 - self.conf_ema_m) * pseudo_label
            return None

    class SupConLoss(nn.Module):
        def __init__(self, temperature=0.07, base_temperature=0.07):
            super().__init__()
            self.temperature = temperature
            self.base_temperature = base_temperature
        def forward(self, features, mask=None, batch_size=-1):
            device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
            if mask is not None:
                mask = mask.float().detach().to(device)
                anchor_dot_contrast = torch.div(
                    torch.matmul(features[:batch_size], features.T),
                    self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()
                logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
                mask = mask * logits_mask
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
                mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss = loss.mean()
            else:
                q = features[:batch_size]
                k = features[batch_size:batch_size * 2]
                queue = features[batch_size * 2:]
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                l_neg = torch.einsum('nc,kc->nk', [q, queue])
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                loss = F.cross_entropy(logits, labels)
            return loss

    def __init__(self,input_shape,train_givenY,hparams):
        super(PiCO, self).__init__(input_shape,train_givenY,hparams)
        self.network=self.PiCO_model(self.num_classes,input_shape,hparams=hparams,base_encoder=self.PiCONet)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence
        self.loss_fn = self.partial_loss(label_confidence.float(),self.hparams)
        self.loss_cont_fn = self.SupConLoss()
        self.train_givenY = train_givenY
        self.curr_iter = 0
        self.max_steps = self.hparams['num_epochs']

    def update(self,minibatches):
        x, strong_x, partial_y, _, index = minibatches
        cls_out, features_cont, pseudo_target_cont, score_prot = self.network(x, strong_x, partial_y, self.hparams)
        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        start_upd_prot = self.curr_iter >= self.hparams['prot_start']
        if start_upd_prot:
            self.loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=partial_y)
        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
        else:
            mask = None
        loss_cont = self.loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        loss_cls = self.loss_fn(cls_out, index)
        loss = loss_cls + self.hparams['loss_weight'] * loss_cont
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_fn.set_conf_ema_m(self.curr_iter, self.max_steps)
        self.curr_iter = self.curr_iter + 1
        return {'loss': loss.item()}

    def predict(self,images,):
        return self.network(img_q=images,is_eval=True)


class VALEN(Algorithm):
    """
    VALEN
    Reference: Instance-Dependent Partial Label Learning, NeurIPS 2021.
    """

    class VAE_Bernulli_Decoder(nn.Module):
        def __init__(self, n_in, n_hidden, n_out, keep_prob=1.0) -> None:
            super().__init__()
            self.layer1 = nn.Linear(n_in, n_hidden)
            self.layer2 = nn.Linear(n_hidden, n_out)
            self._init_weight()

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.01)

        def forward(self, inputs):
            h0 = self.layer1(inputs)
            h0 = F.relu(h0)
            x_hat = self.layer2(h0)
            return x_hat

    def num_flat_features(self, input_shape):
        #size = x.size()[1:]
        size = input_shape[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(VALEN, self).__init__(input_shape,train_givenY,hparams)
        self.featurizer_net = networks.Featurizer(input_shape, self.hparams)
        self.classifier_net = networks.Classifier(
            self.featurizer_net.n_outputs,
            self.num_classes)
        self.net = nn.Sequential(self.featurizer_net, self.classifier_net)
        self.enc=copy.deepcopy(self.net)
        #self.num_features = input_shape[1] * input_shape[2] * input_shape[3]
        self.num_features = self.num_flat_features(input_shape)
        self.dec=self.VAE_Bernulli_Decoder(self.num_classes, self.num_features, self.num_features)
        self.optimizer = torch.optim.Adam(
            list(self.net.parameters()) + list(self.enc.parameters()) + list(self.dec.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.warmup_opt=torch.optim.SGD(list(self.net.parameters()), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'], momentum=0.9)
        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        partial_weight = train_givenY.float() / tempY
        self.o_array = partial_weight
        self.feature_extracted= torch.zeros((train_givenY.shape[0], self.featurizer_net.n_outputs))

        self.curr_iter = 0
        self.steps_per_epoch = train_givenY.shape[0] / self.hparams['batch_size']
        self.mat_save_path = hparams['output_dir']

    def partial_loss(self,output1, target, true, eps=1e-12):
        output = F.softmax(output1, dim=1)
        l = target * torch.log(output + eps)
        loss = (-torch.sum(l)) / l.size(0)
        revisedY = target.clone()
        revisedY[revisedY > 0] = 1
        revisedY = revisedY * (output.clone().detach() + eps)
        revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)
        new_target = revisedY
        return loss, new_target

    def alpha_loss(self, alpha, prior_alpha):
        KLD = torch.mvlgamma(alpha.sum(1), p=1) - torch.mvlgamma(alpha, p=1).sum(1) - torch.mvlgamma(prior_alpha.sum(1),
                                                                                                     p=1) + torch.mvlgamma(
            prior_alpha, p=1).sum(1) + ((alpha - prior_alpha) * (
                    torch.digamma(alpha) - torch.digamma(alpha.sum(dim=1, keepdim=True).expand_as(alpha)))).sum(1)
        return KLD.mean()

    def update(self,minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if index.is_cuda else "cpu"
        if self.curr_iter<self.hparams['warm_up']:
            phi, outputs = self.net[0](x),self.net(x)
            self.o_array=self.o_array.to(device)
            L_ce, new_labels = self.partial_loss(outputs, self.o_array[index, :].clone().detach(), None)
            self.o_array[index, :] = new_labels.clone().detach()
            self.warmup_opt.zero_grad()
            L_ce.backward()
            self.warmup_opt.step()
            self.curr_iter=self.curr_iter+1
            return {'loss': torch.tensor(0.0)}
        elif self.hparams['warm_up']<=self.curr_iter and self.curr_iter<self.hparams['warm_up']+self.steps_per_epoch+1:
            self.feature_extracted=self.feature_extracted.to(device)
            self.feature_extracted[index, :] = self.net[0](x).detach()
            self.curr_iter=self.curr_iter+1
            return {'loss': torch.tensor(0.0)}
        elif self.hparams['warm_up']+self.steps_per_epoch+1<self.curr_iter and self.curr_iter<self.hparams['warm_up']+self.steps_per_epoch+2:
            self.enc = copy.deepcopy(self.net)
            adj = self.gen_adj_matrix2(self.feature_extracted.cpu().numpy(), k=self.hparams['knn'],
                                  path=os.path.abspath(self.mat_save_path + "/adj_matrix.npy"))
            self.A = adj.to_dense()
            self.adj = adj.to(device)
            self.prior_alpha = torch.Tensor(1, self.num_classes).fill_(1.0).to(device)
            self.d_array = copy.deepcopy(self.o_array)
            self.curr_iter = self.curr_iter + 1
            return {'loss': torch.tensor(0.0)}
        else:
            outputs = self.net(x)
            alpha = self.enc(x)
            s_alpha = F.softmax(alpha, dim=1)
            revised_alpha = torch.zeros_like(partial_y)
            revised_alpha[self.o_array[index, :] > 0] = 1.0
            s_alpha = s_alpha * revised_alpha
            s_alpha_sum = s_alpha.clone().detach().sum(dim=1, keepdim=True)
            s_alpha = s_alpha / s_alpha_sum + 1e-2
            L_d, new_d = self.partial_loss(alpha, self.o_array[index, :], None)
            alpha = torch.exp(alpha / 4)
            alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
            L_alpha = self.alpha_loss(alpha, self.prior_alpha)
            dirichlet_sample_machine = torch.distributions.dirichlet.Dirichlet(s_alpha)
            d = dirichlet_sample_machine.rsample()
            x_hat = self.dec(d.float())
            x_hat = x_hat.view(x.shape)
            A_hat = F.softmax(self.dot_product_decode(d.float()), dim=1)
            L_recx = 0.01 * F.mse_loss(x_hat, x)
            #L_recy = 0.01 * F.binary_cross_entropy_with_logits(d, partial_y)
            L_recy = 0.01 * F.binary_cross_entropy_with_logits(d, partial_y.float())
            L_recA = F.mse_loss(A_hat, self.A.to(device)[index, :][:, index].to(device))
            L_rec = L_recx + L_recy + L_recA
            L_o, new_o = self.partial_loss(outputs, self.d_array[index, :], None)
            L = self.hparams['alpha'] * L_rec + self.hparams['beta'] * L_alpha + self.hparams['gamma'] * L_d + self.hparams['theta'] * L_o
            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
            new_d = self.revised_target(d, new_d)
            new_d = self.hparams['correct'] * new_d + (1 - self.hparams['correct']) * self.o_array[index, :]
            self.d_array[index, :] = new_d.clone().detach()
            self.o_array[index, :] = new_o.clone().detach()
            self.curr_iter = self.curr_iter + 1
            return {'loss': L.item()}

    def predict(self, x):
        return self.net(x)

    def revised_target(self,output, target):
        revisedY = target.clone()
        revisedY[revisedY > 0] = 1
        revisedY = revisedY * (output.clone().detach())
        revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)
        new_target = revisedY

        return new_target
    def dot_product_decode(self,Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def adj_normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def gen_adj_matrix2(self, X, k=10, path=""):
        if os.path.exists(path):
            print("Found adj matrix file and Load.")
            adj_m = np.load(path)
            print("Adj matrix Finished.")
        else:
            print("Not Found adj matrix file and Compute.")
            dm = euclidean_distances(X, X)
            adj_m = np.zeros_like(dm)
            row = np.arange(0, X.shape[0])
            dm[row, row] = np.inf
            for _ in range(0, k):
                col = np.argmin(dm, axis=1)
                dm[row, col] = np.inf
                adj_m[row, col] = 1.0
            np.save(path, adj_m)
            print("Adj matrix Finished.")
        adj_m = sp.coo_matrix(adj_m)
        adj_m = self.adj_normalize(adj_m + sp.eye(adj_m.shape[0]))
        adj_m = self.sparse_mx_to_torch_sparse_tensor(adj_m)
        return adj_m

## Complementary-label learning algorithms

class PC(Algorithm):
    """
    PC
    Reference: Learning from Complementary Labels, NIPS 2017.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(PC, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        total_idxes, comp_labels = torch.where(partial_y == 0)
        K = partial_y.shape[1]
        outputs = self.predict(x)[total_idxes]
        loss = self.pc_loss(outputs, K, comp_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def pc_loss(self, f, K, labels):
        sigmoid = nn.Sigmoid()
        fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
        loss_matrix = sigmoid( -1. * (f - fbar)) # multiply -1 for "complementary"
        M1, M2 = K*(K-1)/2, K-1
        pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
        return pc_loss

    def predict(self, x):
        return self.network(x)[0]

class Forward(Algorithm):
    """
    Forward
    Reference: Learning with Biased Complementary Labels, ECCV 2018.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(Forward, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if partial_y.is_cuda else "cpu"
        K = partial_y.shape[1]
        total_idxes, comp_labels = torch.where(partial_y == 0)
        outputs = self.predict(x)[total_idxes]
        loss = self.forward_loss(f=outputs, K=K, labels=comp_labels, device=device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def forward_loss(self, f, K, labels, device):
        Q = torch.ones(K,K) * 1/(K-1)
        Q = Q.to(device)
        for k in range(K):
            Q[k,k] = 0
        q = torch.mm(F.softmax(f, 1), Q)
        return F.nll_loss(q.log(), labels.long())

    def predict(self, x):
        return self.network(x)[0]

class NN(Algorithm):
    """
    NN
    Reference: Complementary-Label Learning for Arbitrary Losses and Models, ICML 2019.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(NN, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.ccp = self.class_prior(train_givenY)

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if index.is_cuda else "cpu"
        total_idxes, comp_labels = torch.where(partial_y == 0)
        K = partial_y.shape[1]
        outputs = self.predict(x)[total_idxes]
        loss = self.non_negative_loss(f=outputs, K=K, labels=comp_labels, ccp=self.ccp, beta=self.hparams['beta'], device=device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def non_negative_loss(self, f, K, labels, ccp, beta, device):
        ccp = torch.from_numpy(ccp).float().to(device)
        neglog = -F.log_softmax(f, dim=1)
        loss_vector = torch.zeros(K, requires_grad=True).to(device)
        temp_loss_vector = torch.zeros(K).to(device)
        for k in range(K):
            idx = labels == k
            if torch.sum(idx).item() > 0:
                idxs = idx.bool().view(-1,1).repeat(1,K)
                neglog_k = torch.masked_select(neglog, idxs).view(-1,K)
                temp_loss_vector[k] = -(K-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
                loss_vector = loss_vector + torch.mul(ccp, torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max 
        loss_vector = loss_vector + temp_loss_vector
        count = np.bincount(labels.data.cpu()).astype('float')
        while len(count) < K:
            count = np.append(count, 0) # when largest label is below K, bincount will not take care of them
        loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)-beta), 1)
        max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
        final_loss = torch.sum(max_loss_vector)
        return final_loss

    def class_prior(self, train_givenY):
        _, complementary_labels = np.where(train_givenY == 0)
        return np.bincount(complementary_labels) / len(complementary_labels)

    def predict(self, x):
        return self.network(x)[0]

class GA(Algorithm):
    """
    GA
    Reference: Complementary-Label Learning for Arbitrary Losses and Models, ICML 2019.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(GA, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.ccp = self.class_prior(train_givenY)

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if index.is_cuda else "cpu"
        total_idxes, comp_labels = torch.where(partial_y == 0)
        K = partial_y.shape[1]
        outputs = self.predict(x)[total_idxes]
        loss, loss_vector = self.assump_free_loss(f=outputs, K=K, labels=comp_labels, ccp=self.ccp, device=device)
        self.optimizer.zero_grad()
        if torch.min(loss_vector).item() < 0:
            loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)), 1)
            min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
            loss = torch.sum(min_loss_vector)
            loss.backward()
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    p.grad = -1*p.grad
        else:
            loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def non_negative_loss(self, f, K, labels, ccp, beta, device):
        ccp = torch.from_numpy(ccp).float().to(device)
        neglog = -F.log_softmax(f, dim=1)
        loss_vector = torch.zeros(K, requires_grad=True).to(device)
        temp_loss_vector = torch.zeros(K).to(device)
        for k in range(K):
            idx = labels == k
            if torch.sum(idx).item() > 0:
                idxs = idx.bool().view(-1,1).repeat(1,K)
                neglog_k = torch.masked_select(neglog, idxs).view(-1,K)
                temp_loss_vector[k] = -(K-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
                loss_vector = loss_vector + torch.mul(ccp, torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max 
        loss_vector = loss_vector + temp_loss_vector
        count = np.bincount(labels.data.cpu()).astype('float')
        while len(count) < K:
            count = np.append(count, 0) # when largest label is below K, bincount will not take care of them
        loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)-beta), 1)
        max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
        final_loss = torch.sum(max_loss_vector)
        return final_loss, torch.mul(torch.from_numpy(count).float().to(device), loss_vector)

    def assump_free_loss(self, f, K, labels, ccp, device):
        """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is negative inf."""
        return self.non_negative_loss(f=f, K=K, labels=labels, ccp=ccp, beta=np.inf, device=device)

    def class_prior(self, train_givenY):
        _, complementary_labels = np.where(train_givenY == 0)
        return np.bincount(complementary_labels) / len(complementary_labels)

    def predict(self, x):
        return self.network(x)[0]

class SCL_EXP(Algorithm):
    """
    SCL-EXP
    Reference: Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels, ICML 2020.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(SCL_EXP, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        total_idxes, comp_labels = torch.where(partial_y == 0)
        outputs = self.predict(x)[total_idxes]
        loss = self.SCL_EXP_loss(f=outputs, labels=comp_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def SCL_EXP_loss(self, f, labels):
        sm_outputs = F.softmax(f, dim=1)
        loss = -F.nll_loss(sm_outputs.exp(), labels.long())
        return loss

    def predict(self, x):
        return self.network(x)[0]

class SCL_NL(Algorithm):
    """
    SCL-NL
    Reference: Unbiased Risk Estimators Can Mislead: A Case Study of Learning with Complementary Labels, ICML 2020.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(SCL_NL, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        total_idxes, comp_labels = torch.where(partial_y == 0)
        outputs = self.predict(x)[total_idxes]
        loss = self.SCL_NL_loss(f=outputs, labels=comp_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def SCL_NL_loss(self, f, labels):
        p = (1 - F.softmax(f, dim=1) + 1e-6).log()
        loss = F.nll_loss(p, labels.long())
        return loss

    def predict(self, x):
        return self.network(x)[0]

class L_W(Algorithm):
    """
    L-W
    Reference: Discriminative Complementary-Label Learning with Weighted Loss, ICML 2021.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(L_W, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        total_idxes, comp_labels = torch.where(partial_y == 0)
        outputs = self.predict(x)[total_idxes]
        K = partial_y.shape[1]
        loss = self.w_loss(f=outputs, K=K, labels=comp_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def non_k_softmax_loss(self, f, K, labels):
        Q_1 = 1 - F.softmax(f, 1)
        Q_1 = F.softmax(Q_1, 1)
        labels = labels.long()
        return F.nll_loss(Q_1.log(), labels.long())

    def w_loss(self, f, K, labels):

        loss_class = self.non_k_softmax_loss(f=f, K=K, labels=labels)
        loss_w = self.w_loss_p(f=f, K=K, labels=labels)
        final_loss = loss_class + loss_w
        return final_loss

    def w_loss_p(self, f, K, labels):
        Q_1 = 1-F.softmax(f, 1)
        Q = F.softmax(Q_1, 1)
        q = torch.tensor(1.0) / torch.sum(Q_1, dim=1)
        q = q.view(-1, 1).repeat(1, K)
        w = torch.mul(Q_1, q)  # weight
        w_1 = torch.mul(w, Q.log())
        return F.nll_loss(w_1, labels.long())

    def predict(self, x):
        return self.network(x)[0]

class OP_W(Algorithm):
    """
    OP-W
    Reference: Consistent Complementary-Label Learning via Order-Preserving Losses, AISTATS 2023.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(OP_W, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        total_idxes, comp_labels = torch.where(partial_y == 0)
        outputs = self.predict(x)[total_idxes]
        K = partial_y.shape[1]
        loss = self.OP_W_loss(f=outputs, K=K, labels=comp_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def OP_W_loss(self, f, K, labels):
        Q_1 = F.softmax(f, 1)+1e-18
        Q_2 = F.softmax(-f, 1)+1e-18
        w_ = torch.div(1, Q_2)

        w_ = w_ + 1
        w = F.softmax(w_,1)

        w = torch.mul(Q_1,w)+1e-6
        w_1 = torch.mul(w, Q_2.log())
        l2 = F.nll_loss(w_1, labels.long())
        return l2

    def predict(self, x):
        return self.network(x)[0]

class FREDIS(Algorithm):
    """
    FREDIS
    Reference: FREDIS: A Fusion Framework of Refinement and Disambiguation for Unreliable Partial Label Learning, ICML 2023.
    """

    def __init__(self, model, input_shape, train_givenY, hparams):
        super(FREDIS, self).__init__(model, input_shape, train_givenY, hparams)

        self.network = model
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # self.train_givenY = torch.from_numpy(train_givenY)
        tempY = self.train_givenY.sum(dim=1).unsqueeze(1).repeat(1, self.train_givenY.shape[1])
        label_confidence = self.train_givenY/tempY
        self.confidence = label_confidence.double()
        self.posterior = torch.ones_like(self.confidence).double()
        self.posterior = self.posterior / self.posterior.sum(dim=1, keepdim=True)
        self.steps_per_epoch = train_givenY.shape[0] / self.hparams['batch_size']  
        self.steps_update_interval = int(self.hparams['update_interval'] * self.steps_per_epoch)
        self.curr_iter = 0
        self.theta = self.hparams['theta']
        self.delta = self.hparams['delta']
        self.pre_correction_label_matrix = torch.zeros_like(self.train_givenY)
        self.correction_label_matrix = copy.deepcopy(self.train_givenY)

    def update(self, minibatches):
        x, strong_x, partial_y, _, index = minibatches
        device = "cuda" if index.is_cuda else "cpu"
        self.confidence = self.confidence.to(device)
        self.posterior = self.posterior.to(device)
        self.correction_label_matrix = self.correction_label_matrix.to(device)
        self.pre_correction_label_matrix = self.pre_correction_label_matrix.to(device)
        consistency_criterion = nn.KLDivLoss(reduction='batchmean').to(device)
        batch_outputs = self.predict(x)
        y_pred_aug0_probas_log = torch.log_softmax(batch_outputs, dim=-1)
        consist_loss = consistency_criterion(y_pred_aug0_probas_log, self.confidence[index].float())
        #print(y_pred_aug0_probas_log.dtype)
        #print(self.confidence.dtype)
        #print(batch_outputs.dtype)
        super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(batch_outputs, dim=1)) * (1 - partial_y), dim=1))
        loss = float(self.hparams['lam']) * consist_loss + float(self.hparams['alpha']) * super_loss
        #print(super_loss.dtype)
        #print(consist_loss.dtype)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.confidence_update(x, partial_y, index)
        self.posterior_update(x, index)
        if self.curr_iter % self.steps_update_interval == 0:
            pred, _ = torch.max(self.posterior, dim=1, keepdim=True)
            tmp_diff = pred - self.posterior            
            self.pre_correction_label_matrix = copy.deepcopy(self.correction_label_matrix)
            non_change_matrix = copy.deepcopy(self.pre_correction_label_matrix)
            non_change_matrix[tmp_diff < self.theta] = 1
            non_change = torch.sum(torch.not_equal(self.pre_correction_label_matrix, non_change_matrix))
            if non_change > self.hparams['change_size']:
                row, col = torch.where(tmp_diff < self.theta)
                idx_list = [ i for i in range(0, len(row))]
                random.shuffle(idx_list)
                non_row, non_col = row[idx_list[0:self.hparams['change_size']]], col[idx_list[0:self.hparams['change_size']]]
                non_change = self.hparams['change_size']
                self.correction_label_matrix[non_row, non_col] = 1
            else:
                self.correction_label_matrix[tmp_diff < self.theta] = 1

            can_change_matrix = copy.deepcopy(self.pre_correction_label_matrix)
            can_change_matrix[tmp_diff > self.delta] = 0
            can_change = torch.sum(torch.not_equal(self.pre_correction_label_matrix, can_change_matrix))
            while can_change < non_change * self.hparams['times']:
                # delta *= (1 - dec)
                self.delta = self.delta - self.hparams['dec']
                can_change_matrix = copy.deepcopy(self.pre_correction_label_matrix)
                can_change_matrix[tmp_diff > self.delta] = 0
                can_change = torch.sum(torch.not_equal(self.pre_correction_label_matrix, can_change_matrix))
            if can_change > self.hparams['change_size'] * self.hparams['times']:
                row, col = torch.where(tmp_diff > self.delta)
                idx_list = [ i for i in range(0, len(row))]
                random.shuffle(idx_list)
                can_row, can_col = row[idx_list[0: self.hparams['change_size'] * self.hparams['times']]], col[idx_list[0: self.hparams['change_size'] * self.hparams['times']]]
                can_change = self.hparams['change_size'] * self.hparams['times']
                self.correction_label_matrix[can_row, can_col] = 0
            else:
                self.correction_label_matrix[tmp_diff > self.delta] = 0
            for i in range(len(self.correction_label_matrix)):
                if self.correction_label_matrix[i].sum == 0:
                    self.correction_label_matrix[i] = copy.deepcopy(self.pre_correction_label_matrix[i])

            tmp_label_matrix = self.posterior * (self.correction_label_matrix + 1e-12)
            self.confidence = tmp_label_matrix / tmp_label_matrix.sum(dim=1).repeat(tmp_label_matrix.size(1), 1).transpose(0, 1)

            change = torch.sum(torch.not_equal(self.pre_correction_label_matrix, self.correction_label_matrix))  

            if self.theta < 0.9 and self.delta > 0.1:
                if change < self.hparams['change_size']:
                    self.theta += self.hparams['inc']
                    self.delta -= self.hparams['dec']
        self.curr_iter = self.curr_iter + 1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)[0]

    def confidence_update(self, batchX, batchY, batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            self.confidence[batch_index, :] = (temp_un_conf * batchY).double() # un_confidence stores the weight of each example
            base_value = self.confidence.sum(dim=1).unsqueeze(1).repeat(1, self.confidence.shape[1])
            self.confidence = self.confidence / base_value

    def posterior_update(self, batchX, batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            self.posterior[batch_index, :] = torch.softmax(batch_outputs, dim=-1).double()

class PiCO_plus(PiCO):
    """
    PiCO_plus: PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning, TPAMI 2024.
    """

    class PiCO_plus_model(PiCO.PiCO_model):
        def __init__(self, num_classes, input_shape, hparams, base_encoder):
            super().__init__(num_classes,input_shape,hparams, base_encoder)
            self.register_buffer("queue_rel", torch.zeros(hparams['moco_queue'], dtype=torch.bool))
        @torch.no_grad()
        def _dequeue_and_enqueue(self, keys, labels, is_rel, hparams):
            batch_size = is_rel.shape[0]
            ptr = int(self.queue_ptr)
            self.queue_rel[ptr:ptr + batch_size] = is_rel
            super()._dequeue_and_enqueue(keys, labels, hparams)

        def forward(self, img_q, im_k=None, Y_ori=None, Y_cor=None, is_rel=None, hparams=None, is_eval=False, ):
            output, q = self.encoder_q(img_q)
            if is_eval:
                return output

            batch_weight = is_rel.float()
            with torch.no_grad():
                predicetd_scores = torch.softmax(output, dim=1)
                _, within_max_cls = torch.max(predicetd_scores * Y_ori, dim=1)
                _, all_max_cls = torch.max(predicetd_scores, dim=1)
                pseudo_labels_b = batch_weight * within_max_cls + (1 - batch_weight) * all_max_cls
                pseudo_labels_b = pseudo_labels_b.long()
                prototypes = self.prototypes.clone().detach()
                logits_prot = torch.mm(q, prototypes.t())
                score_prot = torch.softmax(logits_prot, dim=1)
                _, within_max_cls_ori = torch.max(predicetd_scores * Y_ori, dim=1)
                distance_prot = - (q * prototypes[within_max_cls_ori]).sum(dim=1)
                with torch.no_grad():
                    for feat, label in zip(q[is_rel], pseudo_labels_b[is_rel]):
                        self.prototypes[label] = self.prototypes[label] * hparams['proto_m'] + (1 - hparams['proto_m']) * feat
                self.prototypes = F.normalize(self.prototypes, p=2, dim=1).detach()
                self._momentum_update_key_encoder(hparams)
                _, k = self.encoder_k(im_k)
            features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
            pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
            is_rel_queue = torch.cat((is_rel, is_rel, self.queue_rel.clone().detach()), dim=0)
            self._dequeue_and_enqueue(k, pseudo_labels_b, is_rel, hparams)
            return output, features, pseudo_labels, score_prot, distance_prot, is_rel_queue

    class partial_loss(nn.Module):
        def __init__(self, confidence, hparams, conf_ema_m=0.99):
            super().__init__()
            self.confidence = confidence
            self.conf_ema_m = conf_ema_m
            self.num_class = confidence.shape[1]
            self.conf_ema_range = [float(item) for item in hparams['conf_ema_range'].split(',')]

        def set_conf_ema_m(self, epoch, total_epochs):
            start = self.conf_ema_range[0]
            end = self.conf_ema_range[1]
            self.conf_ema_m = 1. * epoch / total_epochs * (end - start) + start

        def forward(self, outputs, index, is_rel=None):
            device = "cuda" if outputs.is_cuda else "cpu"
            self.confidence = self.confidence.to(device)
            confidence = self.confidence[index, :].to(device)
            loss_vec, _ = self.ce_loss(outputs, confidence)
            if is_rel is None:
                average_loss = loss_vec.mean()
            else:
                average_loss = loss_vec[is_rel].mean()
            return average_loss

        def ce_loss(self, outputs, targets, sel=None):
            targets = targets.detach()
            logsm_outputs = F.log_softmax(outputs, dim=1)
            final_outputs = logsm_outputs * targets
            loss_vec = - (final_outputs).sum(dim=1)
            if sel is None:
                average_loss = loss_vec.mean()
            else:
                if sel.sum()==0:
                    average_loss=torch.tensor(0.0).cuda()
                else:
                    average_loss = loss_vec[sel].mean()
            return loss_vec, average_loss

        def confidence_update(self, temp_un_conf, batch_index, batchY):
            with torch.no_grad():
                _, prot_pred = (temp_un_conf * batchY).max(dim=1)
                device = (torch.device('cuda') if temp_un_conf.is_cuda else torch.device('cpu'))
                pseudo_label = F.one_hot(prot_pred, self.num_class).float().to(device).detach()
                self.confidence=self.confidence.to(device)
                self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] + (1 - self.conf_ema_m) * pseudo_label
            return None

    class SupConLoss(nn.Module):
        def __init__(self, temperature=0.07, base_temperature=0.07):
            super().__init__()
            self.temperature = temperature
            self.base_temperature = base_temperature
        def forward(self, features, mask=None, batch_size=-1, weights=None):
            device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
            if mask is not None:
                mask = mask.float().detach().to(device)
                anchor_dot_contrast = torch.div(
                    torch.matmul(features[:batch_size], features.T),
                    self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()
                logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
                mask = mask * logits_mask
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
                mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                if weights is None:
                    loss = loss.mean()
                else:
                    weights = weights.detach()
                    loss = (loss * weights).mean()
            else:
                q = features[:batch_size]
                k = features[batch_size:batch_size * 2]
                queue = features[batch_size * 2:]
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                l_neg = torch.einsum('nc,kc->nk', [q, queue])
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
                loss = F.cross_entropy(logits, labels)
            return loss



    def reliable_set_selection(self, hparams, epoch, sel_stats):
        dist = sel_stats['dist']
        n = dist.shape[0]
        device = (torch.device('cuda') if dist.is_cuda else torch.device('cpu'))
        is_rel = torch.zeros(n).bool().to(device)
        sorted_idx = torch.argsort(dist)
        chosen_num = int(n * hparams['pure_ratio'])
        is_rel[sorted_idx[:chosen_num]] = True
        sel_stats['is_rel'] = is_rel

    def __init__(self,input_shape,train_givenY,hparams):
        super(PiCO_plus, self).__init__(input_shape,train_givenY,hparams)
        self.network=self.PiCO_plus_model(self.num_classes,input_shape,hparams=hparams,base_encoder=self.PiCONet)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.num_instance=train_givenY.shape[0]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sel_stats = {'dist': torch.zeros(self.num_instance).to(device),'is_rel': torch.ones(self.num_instance).bool().to(device)}

    def update(self, minibatches):
        if self.curr_iter >= self.hparams['prot_start']:
            self.reliable_set_selection(self.hparams, self.curr_iter, self.sel_stats)
        start_upd_prot = self.curr_iter >= self.hparams['prot_start']

        x, strong_x, partial_y, _, index = minibatches
        device = (torch.device('cuda') if x.is_cuda else torch.device('cpu'))
        is_rel = self.sel_stats['is_rel'][index]
        batch_weight = is_rel.float()
        cls_out, features_cont, pseudo_labels, score_prot, distance_prot, is_rel_queue = self.network(x, strong_x, partial_y, Y_cor=None, is_rel=is_rel, hparams=self.hparams)
        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_labels.contiguous().view(-1, 1)


        if start_upd_prot:
            self.loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=partial_y)
        if start_upd_prot:
            mask_all = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().to(device)
            loss_cont_all = self.loss_cont_fn(features=features_cont, mask=mask_all, batch_size=batch_size, weights=None)
            mask = copy.deepcopy(mask_all).detach()
            mask = batch_weight.unsqueeze(1).repeat(1, mask.shape[1]) * mask
            mask = is_rel_queue.view(1, -1).repeat(mask.shape[0], 1) * mask
            if self.curr_iter >= self.hparams['knn_start']:
                cosine_corr = features_cont[:batch_size] @ features_cont.T
                _, kNN_index = torch.topk(cosine_corr, k=self.hparams['chosen_neighbors'], dim=-1, largest=True)
                mask_kNN = torch.scatter(torch.zeros(mask.shape).to(device), 1, kNN_index, 1)
                mask[~is_rel] = mask_kNN[~is_rel]
            mask[:, batch_size:batch_size * 2] = ((mask[:, batch_size:batch_size * 2] + torch.eye(batch_size).to(device)) > 0).float()
            mask[:, :batch_size] = ((mask[:, :batch_size] + torch.eye(batch_size).to(device)) > 0).float()
            if self.curr_iter >= self.hparams['knn_start']:
                weights = self.hparams['loss_weight'] * batch_weight + self.hparams['ur_weight'] * (1 - batch_weight)
                loss_cont_rel_knn = self.loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size,weights=weights)
            else:
                loss_cont_rel_knn = self.loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size, weights=None)
            loss_cont = loss_cont_rel_knn + self.hparams['ur_weight'] * loss_cont_all
            loss_cls = self.loss_fn(cls_out, index, is_rel)
            sp_temp_scale = score_prot ** (1 / self.hparams['temperature_guess'])
            targets_guess = sp_temp_scale / sp_temp_scale.sum(dim=1, keepdim=True)
            _, loss_cls_ur = self.loss_fn.ce_loss(cls_out, targets_guess, sel=~is_rel)
            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            pseudo_label = self.loss_fn.confidence[index]
            pseudo_label[~is_rel] = targets_guess[~is_rel]
            idx = torch.randperm(x.size(0))
            X_w_rand = x[idx]
            pseudo_label_rand = pseudo_label[idx]
            X_w_mix = l * x + (1 - l) * X_w_rand
            pseudo_label_mix = l * pseudo_label + (1 - l) * pseudo_label_rand
            logits_mix, _ = self.network.encoder_q(X_w_mix)
            _, loss_mix = self.loss_fn.ce_loss(logits_mix, targets=pseudo_label_mix)
            loss_cls = loss_mix + self.hparams['cls_weight'] * loss_cls + self.hparams['ur_weight'] * loss_cls_ur
            loss = loss_cls + loss_cont
        else:
            loss_cls = self.loss_fn(cls_out, index, is_rel=None)
            loss_cont = self.loss_cont_fn(features=features_cont, mask=None, batch_size=batch_size)
            loss = loss_cls + self.hparams['loss_weight'] * loss_cont

        self.sel_stats['dist'][index] = copy.deepcopy(distance_prot.clone().detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_fn.set_conf_ema_m(self.curr_iter, self.max_steps)
        return {'loss': loss.item()}

class ALIM(PiCO):
    """
    ALIM
    Reference: ALIM: Adjusting Label Importance Mechanism for Noisy Partial Label Learning, NeurIPS 2023
    """


    class partial_loss(nn.Module):
        def __init__(self, confidence, hparams, conf_ema_m=0.99):
            super().__init__()
            self.confidence = confidence
            self.init_conf = confidence.detach()
            self.conf_ema_m = conf_ema_m
            self.conf_ema_range = [float(item) for item in hparams['conf_ema_range'].split(',')]
        def set_conf_ema_m(self, epoch, total_epochs):
            start = self.conf_ema_range[0]
            end = self.conf_ema_range[1]
            self.conf_ema_m = 1. * epoch /total_epochs * (end - start) + start
        def forward(self, outputs, index):
            device = "cuda" if outputs.is_cuda else "cpu"
            self.confidence=self.confidence.to(device)
            logsm_outputs = F.log_softmax(outputs, dim=1)
            final_outputs = logsm_outputs * self.confidence[index, :]
            average_loss = - ((final_outputs).sum(dim=1)).mean()
            return average_loss
        def confidence_update(self, temp_un_conf, batch_index, batchY, piror):
            with torch.no_grad():
                _, prot_pred = (temp_un_conf * (batchY + piror * (1 - batchY))).max(dim=1) # ALIM
                pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
                self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] \
                                                  + (1 - self.conf_ema_m) * pseudo_label
            return None


    def __init__(self,input_shape,train_givenY,hparams):
        super(ALIM, self).__init__(input_shape,train_givenY,hparams)
        self.network=self.PiCO_model(self.num_classes,input_shape,hparams=hparams,base_encoder=self.PiCONet)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # train_givenY = torch.from_numpy(train_givenY)
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence
        self.loss_fn = self.partial_loss(label_confidence.float(),self.hparams)
        self.loss_cont_fn = self.SupConLoss()
        self.train_givenY = train_givenY
        self.piror = 0
        self.curr_iter = 0
        self.margin = []
        self.steps_per_epoch = train_givenY.shape[0] / self.hparams['batch_size']

    def update(self,minibatches1, minibatches2):
        if self.curr_iter % self.steps_per_epoch == 0:
            if self.curr_iter >= self.hparams['start_epoch'] * self.steps_per_epoch:
                self.piror = sorted(self.margin)[int(len(self.margin)*self.hparams['noise_rate'])]
            self.margin = []

        x, strong_x, partial_y, _, index = minibatches1
        _, x2, _, _, _, _, index2 = minibatches2
        cls_out, features_cont, pseudo_target_cont, score_prot = self.network(x, strong_x, partial_y, self.hparams)
        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        start_upd_prot = self.curr_iter >= self.hparams['prot_start']
        if start_upd_prot:
            self.loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=partial_y, piror=self.piror)
        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
        else:
            mask = None
        loss_cont = self.loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        loss_cls = self.loss_fn(cls_out, index)

        lam = np.random.beta(self.hparams['mixup_alpha'], self.hparams['mixup_alpha'])
        lam = max(lam, 1-lam)
        pseudo_label_1 = self.loss_fn.confidence[index]
        pseudo_label_2 = self.loss_fn.confidence[index2]
        X_w_mix = lam * x  + (1 - lam) * x2      
        pseudo_label_mix = lam * pseudo_label_1 + (1 - lam) * pseudo_label_2
        logits_mix, _ = self.network.encoder_q(X_w_mix)
        pred_mix = torch.softmax(logits_mix, dim=1)
        loss_mixup = - ((torch.log(pred_mix) * pseudo_label_mix).sum(dim=1)).mean()

        loss = loss_cls + self.hparams['loss_weight'] * loss_cont + self.hparams['loss_weight_mixup'] * loss_mixup
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_fn.set_conf_ema_m(self.curr_iter, self.max_steps)
        self.margin += ((torch.max(score_prot*partial_y, 1)[0])/(1e-9+torch.max(score_prot*(1-partial_y), 1)[0])).tolist()
        return {'loss': loss.item()}