import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from torch.nn import CrossEntropyLoss

#custom module
from models import get_models
from dataset import AdainDataset
from utils import adjust_lr, cosine_decay_with_warmup

dir_content = r'data\real_A' #訓練集的圖片所在路徑 榮總圖片
dir_truth = r'data\train_mask' #訓練集的真實label所在路徑
dir_style = r'data\fake_B' 

dir_checkpoint = r'log\train1_adain_onlyCE' #儲存模型的權重檔所在路徑
# load_path = r'weights\in\data10000\bestmodel.pth'

os.makedirs(dir_checkpoint,exist_ok=False)

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet on images and target masks')
    parser.add_argument('--image_channel','-i',type=int, default=1,dest='in_channel',help="channels of input images")
    parser.add_argument('--total_epoch','-e',type=int,default=50,metavar='E',help='times of training model')
    parser.add_argument('--warmup_epoch',type=int,default=0,help='warm up the student model')
    parser.add_argument('--batch','-b',type=int,dest='batch_size',default=1, help='Batch size')
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')
    parser.add_argument('--init_lr','-r',type = float, default=2e-2,help='initial learning rate of model')
    parser.add_argument('--device', type=str,default='cuda:0',help='training on cpu or gpu')
    # parser.add_argument('--loss', type=str,default='dice_loss',help='loss metric, options: [kl_divergence, cross_entropy, dice_loss]')
    parser.add_argument('--model', type=str,default='in_unet',help='models, option: bn_unet, in_unet')

    return parser.parse_args()

def main():
    args = get_args()
    trainingDataset = AdainDataset(content_dir = dir_content, truth_dir=dir_truth,style_dir=dir_style)

    #設置 log
    # ref: https://shengyu7697.github.io/python-logging/
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(dir_checkpoint,"log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    ###################################################
    net = get_models(model_name=args.model, is_cls=True,args=args)
    
    # pretrained_model_param_dict = torch.load(load_path)
    # net.load_state_dict(pretrained_model_param_dict)
    
    logging.info(net)
    
    optimizer = torch.optim.Adam(net.parameters(),lr = args.init_lr,betas=(0.9,0.999))
    ##紀錄訓練的一些參數配置
    logging.info(f'''
    =======================================
    
    dir_content: {dir_content}
    dir_truth: {dir_truth}
    dir_style: {dir_style}
    dir_checkpoint: {dir_checkpoint}
    args: 
    {args}
    =======================================
    ''')
    try:
        training(net = net,
                optimizer = optimizer,
                dataset = trainingDataset,
                args=args,
                save_checkpoint= True,)
                
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

    return

def training(net,
             optimizer,
             dataset,
             args,
             save_checkpoint: bool = True):

    arg_loader = dict(batch_size = args.batch_size, num_workers = 0)
    train_loader = DataLoader(dataset,shuffle = False, **arg_loader)
    device = torch.device( args.device if torch.cuda.is_available() else 'cpu')
    #Initial logging
    logging.info(f'''Starting training:
        model:           {args.model}
        Epochs:          {args.total_epoch}
        warm up epoch:   {args.warmup_epoch}
        Batch size:      {args.batch_size}
        Loss metirc:     crossentropy
        Training size:   {len(dataset)}
        checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    net.to(device)
    # loss_fn = Distribution_loss()
    # loss_fn.set_metric(args.loss)
    loss_fn = CrossEntropyLoss()
    #begin to train model
    epoch_losses = []
    for i in range(1, args.total_epoch+1):
        net.train()
        epoch_loss = 0
        # adjust the learning rate
        lr = cosine_decay_with_warmup(current_iter=i,total_iter=args.total_epoch,warmup_iter=args.warmup_epoch,base_lr=args.init_lr)
        adjust_lr(optimizer,lr)

        for imgs, truthes, style_imgs in tqdm(train_loader):

            imgs = imgs.to(dtype=torch.float32, device = device)
            truthes = truthes.to(device = device)
            style_imgs = style_imgs.to(dtype=torch.float32, device = device)
            logits = net(imgs,style_imgs)

            loss = loss_fn(logits, truthes.squeeze(1)) # truthes 去掉通道軸
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f'Training loss: {epoch_loss:6.4f} at epoch {i}.')
        epoch_losses.append(epoch_loss)

        if (save_checkpoint) :
            torch.save(net.state_dict(), os.path.join(dir_checkpoint,f'unet_{i}.pth'))
            logging.info(f'Model saved at epoch {i}.')
        
    min_loss_at = torch.argmin(torch.tensor(epoch_losses)).item() + 1 
    logging.info(f'min Training loss at epoch {min_loss_at}.')
            
    return

class Distribution_loss(torch.nn.Module):
    """p is target distribution and q is predict distribution"""
    def __init__(self):
        super(Distribution_loss, self).__init__()
        self.metric = self.set_metric()

    def kl_divergence(self,p,q):
        """p and q are both a logit(before softmax function)"""
        prob_p = torch.softmax(p,dim=1)
        kl = (prob_p * torch.log_softmax(p,dim=1)) - (prob_p * torch.log_softmax(q,dim=1))
        # print(f"p*torch.log(p) is {torch.sum(p*torch.log(p))}")
        # print(f"p*torch.log(q) is {torch.sum(p*torch.log(q))}")
        # print(f"mean kl divergence: {torch.sum(kl) / (kl.shape[0]*kl.shape[-1]*kl.shape[-2])}")
        return torch.sum(kl) / (kl.shape[0]*kl.shape[-1]*kl.shape[-2])

    def cross_entropy(self,p,q):
        """p and q are both a logit(before softmax function)""" 
        ce = -torch.softmax(p, dim=1) * torch.log_softmax(q, dim=1)
        # print(f"mean ce: {torch.sum(ce) / (ce.shape[0]*ce.shape[-1]*ce.shape[-2])}")
        return torch.sum(ce) / (ce.shape[0]*ce.shape[-1]*ce.shape[-2])
    
    def dice_loss(self,p,q):
        smooth = 1e-8
        prob_p = torch.softmax(p,dim=1)
        prob_q = torch.softmax(q,dim=1)

        inter = torch.sum(prob_p*prob_q) + smooth
        union = torch.sum(prob_p) + torch.sum(prob_q) + smooth
        loss = 1 - ((2*inter) / union)
        return  loss / p.size(0) # loss除以batch size

    def forward(self,p,q):
        assert p.dim() == 4, f"dimension of target distribution has to be 4, but get {p.dim()}"
        assert p.dim() == q.dim(), f"dimension dismatch between p and q"
        if self.metric == 'kl_divergence':
            return self.kl_divergence(p,q)
        elif self.metric == "cross_entropy":
            return self.cross_entropy(p,q)
        elif self.metric == "dice_loss":
            return self.dice_loss(p,q)
        else:
            raise NotImplementedError("the loss metric has not implemented")
        
    def set_metric(self, metric:str="cross_entropy"):
        if metric in ["kl_divergence", "cross_entropy", "dice_loss"]:
            self.metric = metric
        else:
            raise NotImplementedError(f"the loss metric has not implemented. metric name must be in kl_divergence or cross_entropy")

if __name__ == '__main__':
    main()