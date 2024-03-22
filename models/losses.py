import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_prob = torch.softmax(x,dim=1)
        xs_pos = x_prob[:,1,:,:]
        xs_neg = x_prob[:,0,:,:]

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1) # 富樣本的機率如果高於(1-margin), 將其機率直接調成1，這樣計算loss的話便不會考慮到該樣本。

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma) # 記住富樣本機率是(1-p)，所以對照ASL公式看1-(1-p) = p, ASL公式的p為正樣本的機率
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()
    
class Distribution_loss(torch.nn.Module):
    """p is target distribution and q is predict distribution"""
    def __init__(self,args=None):
        super(Distribution_loss, self).__init__()
        self.metric = self.set_metric()
        self.args = args

    # custom
    # def _cross_entropy(self,p,target):
    #     """p is logit(before softmax function) custom crossentropy"""
    #     target_onehot = torch.zeros_like(p)
    #     target_onehot.scatter_(1,target,1)
    #     ce = -target_onehot * torch.log_softmax(p, dim=1)
    #     return torch.mean(ce)
    
    #pytorch api
    def _cross_entropy(self,p,target):
        """p is logit(before softmax function)"""
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(p,target.squeeze(1)) # 去掉通道軸
        return loss
    
    def _asymmetric_loss(self,p,target, args):
        """p is logit(before softmax function)""" 
        loss_fn = AsymmetricLoss(gamma_pos=args.gamma_pos,gamma_neg=args.gamma_neg,clip=args.clip)
        mean_loss = loss_fn(p,target)
        return mean_loss
    
    def forward(self,p,target):
        assert p.dim() == 4, f"dimension of target distribution has to be 4, but get {p.dim()}"
        if self.metric == "cross_entropy":
            return self._cross_entropy(p,target)
        elif self.metric == "asymmetric_loss":
            return self._asymmetric_loss(p,target,self.args)
        else:
            raise NotImplementedError("the loss metric has not implemented")
        
    def set_metric(self, metric:str="cross_entropy"):
        if metric in ["cross_entropy", "asymmetric_loss"]:
            self.metric = metric
        else:
            raise NotImplementedError(f"the loss metric has not implemented. metric name must be in kl_divergence or cross_entropy")