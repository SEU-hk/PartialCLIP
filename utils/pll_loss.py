import torch
import torch.nn.functional as F
import torch.nn as nn

def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss
    
class CC_Loss(nn.Module):
    def __init__(self):
        super(CC_Loss, self).__init__()
        
    def forward(self, outputs, partialY):
        sm_outputs = F.softmax(outputs, dim=1)
        final_outputs = sm_outputs * partialY
        average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
        return average_loss
    
       
def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = - ((final_outputs).sum(dim=1)).mean()
    return average_loss


class RC_Loss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.confidence = target
        
    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    

def lwc_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio):
    device = outputs.device

    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sm_outputs = F.softmax(outputs, dim=1)

    sig_loss1 = - torch.log(sm_outputs + 1e-8)
    l1 = confidence[index, :] * onezero * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
    l2 = confidence[index, :] * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, average_loss1, lw_weight * average_loss2


class LWC_Loss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.label_confidence = target
        
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
        average_loss = average_loss1 + self.hparams["lw_weight"] * average_loss2
        return average_loss
    
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
    

class Proden_loss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.confidence = target


    def forward(self,output1,index,update_target=True):
        output = F.softmax(output1, dim=1)
        target = self.confidence[index, :]
        l = target * torch.log(output)
        loss = (-torch.sum(l)) / l.size(0)
        if update_target:
            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY = revisedY * output
            revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss


class Proden_loss_RECORDS(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.confidence = target
        self.feat_mean = None


    def forward(self,output1,feat,model,index,update_target=True):
        output = F.softmax(output1, dim=1)
        target = self.confidence[index, :]
        l = target * torch.log(output)
        loss = (-torch.sum(l)) / l.size(0)

        if self.feat_mean is None:
            self.feat_mean = 0.1*feat.detach().mean(0)
        else:
            self.feat_mean = 0.9*self.feat_mean + 0.1*feat.detach().mean(0)

        
        if update_target:
            bias = model.module.fc(self.feat_mean.unsqueeze(0)).detach()
            bias = F.softmax(bias, dim=1)
            logits = output1 - torch.log(bias + 1e-9) 
            output = F.softmax(logits, dim=1)


            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY = revisedY * output
            revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss

class Proden_loss_prior(nn.Module):
    def __init__(self, target,partial_prior):
        super().__init__()
        self.confidence = target
        self.feat_mean = None
        self.partial_prior = torch.tensor(partial_prior).float().cuda()


    def forward(self,output1,index,update_target=True):
        output = F.softmax(output1, dim=1)
        target = self.confidence[index, :]
        l = target * torch.log(output)
        loss = (-torch.sum(l)) / l.size(0)

        
        if update_target:

            logits = output1 - torch.log(self.partial_prior + 1e-9) 
            output = F.softmax(logits, dim=1)


            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY = revisedY * output
            revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss

    def __init__(self, target):
        super().__init__()
        self.confidence = target


    def forward(self,output_w,output_s,index,update_target=True):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(1-pred_w)-torch.log(1-pred_s))
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        loss = sup_loss1 + con_loss
        if update_target:
            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY_s = revisedY * pred_s
            resisedY_w = revisedY * pred_w
            revisedY = revisedY_s * resisedY_w
            # sqr
            revisedY = torch.sqrt(revisedY)
            revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss


class CORR_loss(torch.nn.Module):
    def __init__(self, target):
        super().__init__()
        self.confidence = target


    def forward(self,output_w,output_s,index,update_target=True):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(abs(1-pred_w)+1e-9)-torch.log(abs(1-pred_s)+1e-9))
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        loss = sup_loss1 + con_loss
        if update_target:
            revisedY = target.clone()
            revisedY[revisedY > 0]  = 1
            revisedY_s = revisedY * pred_s
            resisedY_w = revisedY * pred_w
            revisedY = revisedY_s * resisedY_w
            # sqr
            revisedY = torch.sqrt(revisedY)
            revisedY = (revisedY) / (revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)+1e-9)

            new_target = revisedY

            self.confidence[index,:]=new_target.detach()

        return loss

class CORR_loss_RECORDS(nn.Module):
    def __init__(self, target, m = 0.9):
        super().__init__()
        self.confidence = target
        self.init_confidence = target.clone()
        self.feat_mean = None
        self.m = m


    def forward(self,output_w,output_s,feat_w,feat_s,model,index,update_target=True):
        pred_s = F.softmax(output_s, dim=1)
        pred_w = F.softmax(output_w, dim=1)
        target = self.confidence[index, :]
        neg = (target==0).float()
        sup_loss = neg * (-torch.log(abs(1-pred_w)+1e-9)-torch.log(abs(1-pred_s)+1e-9))
        if torch.any(torch.isnan(sup_loss)):
            print("sup_loss:nan")
        sup_loss1 = torch.sum(sup_loss) / sup_loss.size(0)
        con_loss = F.kl_div(torch.log_softmax(output_w,dim=1),target,reduction='batchmean')+F.kl_div(torch.log_softmax(output_s,dim=1),target,reduction='batchmean')
        if torch.any(torch.isnan(con_loss)):
            print("con_loss:nan")
        loss = sup_loss1 + con_loss

        if self.feat_mean is None:
            self.feat_mean = (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)
        else:
            self.feat_mean = self.m*self.feat_mean + (1-self.m)*((feat_w+feat_s)/2).detach().mean(0)

        
        if update_target:
            bias = model.module.fc(self.feat_mean.unsqueeze(0)).detach()
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
            # new_target = torch.where(torch.isnan(new_target), self.init_confidence[index,:].clone(), new_target)

            self.confidence[index,:]=new_target.detach()

        return loss

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
    
    
    
class CORR_RECORDS(nn.Module):
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

            self.confidence[index,:] = self.m * self.confidence[index,:] + (1 - self.m) * new_target.detach()

        return loss

