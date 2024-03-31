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
        
        
def _calculate_means(pred, gt, n_objects:list):
    """pred: bs, n_filters, height, width
       gt: bs, n_instances, height, width
       n_objects: 每張圖片內有幾個物件"""
    assert pred.size(0) == gt.size(0)
    assert pred.size(0) == len(n_objects)
    feat_size = pred.size()
    n_instances = gt.size(1)

    pred_repeated = pred.unsqueeze(2).expand(
        feat_size[0], feat_size[1], n_instances, feat_size[2],feat_size[3])  # bs, n_filters, n_instances, h, w
    
    gt_expanded = gt.unsqueeze(1) # bs, 1 , n_instances, h, w

    fg_masked = pred_repeated * gt_expanded

    means = []
    for i in range(feat_size[0]):
        _n_objects_sample = n_objects[i]
        
        _pred_masked_sample = fg_masked[i, :, : _n_objects_sample] # n_filters, n_instances, h, w
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample] # 1 , n_instances, h, w
        _mean_sample = _pred_masked_sample.sum([2,3]) / _gt_expanded_sample.sum([2,3])  # n_filters, n_instances
        
        means.append(_mean_sample)
    if len(means) == 1:
        means = means[0].unsqueeze(0) # 1, n_filters, n_instances
    else:
        means = torch.stack(means,dim=0) # bs, n_fliters, n_instances

    return means


def calculate_variance_loss(pred, gt, n_objects, delta_v:float=1, norm=2):
    """pred: bs, n_filters(channel), height, width
       gt: bs, n_instances(channel), height, width"""
    assert pred.size(0) == gt.size(0)
    assert pred.size(0) == len(n_objects)
    
    feat_size = pred.size()
    n_instances = gt.size(1)
    means = _calculate_means(pred, gt, n_objects) # bs, n_filters, n_instances

    means = means.unsqueeze(3).unsqueeze(4).expand(feat_size[0], feat_size[1], n_instances, feat_size[2],feat_size[3]) # bs, n_filters, n_instances, height, width
    pred = pred.unsqueeze(2).expand(feat_size[0], feat_size[1], n_instances, feat_size[2],feat_size[3])# bs, n_filters, n_instances, height, width
    gt = gt.unsqueeze(1).expand(feat_size[0], feat_size[1], n_instances, feat_size[2],feat_size[3])# bs, n_filters, n_instances, height, width
    bg_mask = (gt == 0).type(torch.int64)

    fg_var = (torch.clamp(torch.norm((pred - means), norm, 1) - delta_v, min=0.0) ** 2) * gt[:,0, :, :, :] # bs, n_instances, height, width
    bg_var = (torch.clamp(torch.norm((pred - means), norm, 1) - (20*delta_v), min=0.0) ** 2) * bg_mask[:,0, :, :, :] # bs, n_instances, height, width

    fg_var_term = 0.0
    bg_var_term = 0.0
    for i in range(feat_size[0]):
        fg_var_sample = fg_var[i, :n_objects[i]]  # n_instances, height, width
        _gt_sample = gt[i, 0, :n_objects[i]]  # n_instances, height, width

        bg_var_sample = bg_var[i, :n_objects[i]]  # n_instances, height, width
        _bg_sample = bg_mask[i, 0, :n_objects[i]]  # n_instances, height, width

        fg_var_term += torch.sum(fg_var_sample) / torch.sum(_gt_sample)
        bg_var_term += torch.sum(bg_var_sample) / torch.sum(_bg_sample)

    fg_var_term = fg_var_term / feat_size[0] # 越小越好
    bg_var_term = bg_var_term / feat_size[0] # 越大越好
    total_var_loss = ((20*delta_v) / (bg_var_term + 1e-8)) + fg_var_term
    # print(total_var_loss)

    return total_var_loss

# class RBF(nn.Module):

#     def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
#         super().__init__()
#         self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
#         self.bandwidth = bandwidth

#     def get_bandwidth(self, L2_distances):
#         if self.bandwidth is None:
#             n_samples = L2_distances.shape[0]
#             return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

#         return self.bandwidth

#     def forward(self, X):
#         L2_distances = torch.cdist(X, X) ** 2
#         return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


# class MMDLoss(nn.Module):

#     def __init__(self, kernel=RBF()):
#         super().__init__()
#         self.kernel = kernel

#     def forward(self, X, Y):
#         K = self.kernel(torch.vstack([X, Y]))

#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean()
#         XY = K[:X_size, X_size:].mean()
#         YY = K[X_size:, X_size:].mean()
#         return XX - 2 * XY + YY



if __name__ == "__main__":
    p = torch.randn(3,8,16,16)
    gt = torch.randint(0,2,(3,2,16,16))
    n_objects = [1 for i in range(p.size(0))]
    loss = calculate_variance_loss(p,gt,n_objects)
    