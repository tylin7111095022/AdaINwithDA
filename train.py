import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#custom module
from models import get_models
from models.losses import Distribution_loss, calculate_variance_term
from dataset import AdainDataset
from utils import adjust_lr, cosine_decay_with_warmup

dir_content = r'data\real_A' #訓練集的圖片所在路徑 榮總圖片
dir_truth = r'data\train_mask' #訓練集的真實label所在路徑
dir_style = r'data\fake_B' 

dir_checkpoint = r'log\train18_adain_CE_fixencoder_pretrain_instanceHead' #儲存模型的權重檔所在路徑

os.makedirs(dir_checkpoint,exist_ok=False)

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet on images and target masks')
    parser.add_argument('--image_channel','-i',type=int, default=1,dest='in_channel',help="channels of input images")
    parser.add_argument('--total_epoch','-e',type=int,default=50,metavar='E',help='times of training model')
    parser.add_argument('--warmup_epoch',type=int,default=0,help='warm up the student model')
    parser.add_argument('--batch','-b',type=int,dest='batch_size',default=1, help='Batch size')
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')
    parser.add_argument('--loss', type=str,default='cross_entropy',help='loss metric, options: [cross_entropy, asymmetric_loss]')
    parser.add_argument('--init_lr','-r',type = float, default=2e-2,help='initial learning rate of model')
    parser.add_argument('--device', type=str,default='cuda:0',help='training on cpu or gpu')
    parser.add_argument('--pretrain_path', type=str,default=r'weights\in\data10000_100epoch\bestmodel.pth',help='pretrain weight')
    parser.add_argument('--model', type=str,default='in_unet',help='models, option: in_unet')
    parser.add_argument('--pad_mode', action="store_true",default=True, help='unet used crop or pad at skip connection')
    parser.add_argument('--normalize', action="store_true",dest="is_normalize",default=True, help='model normalize layer exist or not')
    parser.add_argument('--styleloss', action="store_true",default=False, help='using style loss during training')
    parser.add_argument('--instanceloss', action="store_true",default=True, help='using instance seg loss during training')
    parser.add_argument('--fix_encoder', action="store_true",default=True, help='fix encoder')
    parser.add_argument('--pair_style', action="store_true",default=True, help='if True then choose a paired style img using cyclegan, or random choose a style img from target domain')
    parser.add_argument('--sup_loss_w',type = float, default=1.0,help='weight of supervise loss')
    parser.add_argument('--style_loss_w',type = float, default=1.0,help='weight of style loss')
    parser.add_argument('--instance_loss_w',type = float, default=1.0,help='weight of style loss')
    parser.add_argument('--gamma_neg',type = float, default=2.,help='ASL param, control loss of negative samples')
    parser.add_argument('--gamma_pos',type = float, default=0.,help='ASL param, control loss of positive samples')
    parser.add_argument('--clip',type = float, default=0.1,help='ASL param, control negative samples prob shift, between 0~1')

    return parser.parse_args()

def main():
    args = get_args()
    trainingDataset = AdainDataset(content_dir = dir_content, truth_dir=dir_truth,style_dir=dir_style,pair_style=args.pair_style)
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
    net.freeze_encoder(is_freeze=args.fix_encoder)
    
    if args.pretrain_path:
        pretrained_model_param_dict = torch.load(args.pretrain_path)
        net.load_state_dict(pretrained_model_param_dict,strict=False)
        print(f"pretrained model: {args.pretrain_path}")
    
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
        Loss metirc:     {args.loss}
        Training size:   {len(dataset)}
        checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    net.to(device)
    net.set_styleloss(args.styleloss)
    loss_fn = Distribution_loss(args=args)
    loss_fn.set_metric(args.loss)

    #begin to train model
    epoch_losses = {"superviseLoss":[], "styleLoss": [], "instanceLoss": []}
    for i in range(1, args.total_epoch+1):
        net.train()
        sup_loss = 0
        style_loss = 0
        instance_loss = 0
        # adjust the learning rate
        lr = cosine_decay_with_warmup(current_iter=i,total_iter=args.total_epoch,warmup_iter=args.warmup_epoch,base_lr=args.init_lr)
        adjust_lr(optimizer,lr)
        count = 0
        for imgs, truthes, style_imgs in tqdm(train_loader):
            imgs = imgs.to(dtype=torch.float32, device = device)
            truthes = truthes.to(device = device)
            style_imgs = style_imgs.to(dtype=torch.float32, device = device)
            logits, pixel_embe, styleloss = net(imgs,style_imgs)
            styleloss = args.style_loss_w * styleloss

            n_objects = [1 for i in range(logits.size(0))]

            instanceloss = calculate_variance_term(pixel_embe,truthes,n_objects) if args.instanceloss else torch.zeros(1,device=device)
            instanceloss = args.instance_loss_w * instanceloss
            superviseloss = loss_fn(logits, truthes)
            superviseloss = args.sup_loss_w * superviseloss

            sup_loss += superviseloss.item()
            style_loss += styleloss.item()
            instance_loss += instanceloss.item()
            total_loss = superviseloss + styleloss + instance_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            count+=1

        logging.info(f'total loss: {(sup_loss+style_loss+instance_loss):6.4f}, supervise loss: {sup_loss:6.4f}, style loss: {style_loss:6.4f}, instance loss: {instance_loss:6.4f} at epoch {i}.')
        epoch_losses["superviseLoss"].append(sup_loss)
        epoch_losses["styleLoss"].append(style_loss)
        epoch_losses["instanceLoss"].append(instance_loss)

        if (save_checkpoint) :
            torch.save(net.state_dict(), os.path.join(dir_checkpoint,f'unet_{i}.pth'))
            logging.info(f'Model saved at epoch {i}.')

    total_losses = torch.zeros(args.total_epoch)
    for _,v in epoch_losses.items():
        total_losses += torch.tensor(v)

    min_loss_at = torch.argmin(total_losses).item() + 1 
    logging.info(f'min Training loss at epoch {min_loss_at}.')
            
    return

if __name__ == '__main__':
    main()