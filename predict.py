import argparse
import os
import cv2
import numpy as np
import torch
import warnings
from PIL import Image
warnings.filterwarnings("ignore")
from tqdm import tqdm
import threading
#custom
from models import get_models
from dataset import GrayDataset
from metric import compute_mIoU, dice_score
from utils import Plotter, GradCam


# test_img_dir = r"D:\tsungyu\chromosome_data\chang\changValSum\images"
# test_truth_dir = r"D:\tsungyu\chromosome_data\chang\changValSum\masks"
test_img_dir = r"D:\tsungyu\chromosome_data\chang\chang_val_4\images"
test_truth_dir = r"D:\tsungyu\chromosome_data\chang\chang_val_4\masks"

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', type=str,default='in_unet',help='models, option: bn_unet, in_unet')
    parser.add_argument('--in_channel','-i',type=int, default=1,help="channels of input images")
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')
    parser.add_argument('--instanceloss', action="store_true",default=True, help='using instance seg loss during training')
    parser.add_argument('--weight', '-w', default=r'log\ASL_varianceLoss\unet_50.pth', metavar='FILE',help='Specify the file in which the model is stored')
    parser.add_argument('--normalize', action="store_true",dest="is_normalize",default=True, help='model normalize layer exist or not')
    parser.add_argument('--pad_mode', action="store_true",default=False, help='unet used crop or pad at skip connection')
    parser.add_argument('--imgpath', '-img',type=str,default=r'A185065_02-01_060318215633_0.png', help='the path of img')
    parser.add_argument('--imgdir', type=str,default=r'', help='the path of directory which imgs saved for predicting')
    parser.add_argument('--outdir', type=str,default=r'testchromosome', help='directory where saved predict imgs')
    parser.add_argument('--eval', action="store_true",default=False, help='calculate miou and dice score')
    
    return parser.parse_args()

def main():
    args = get_args()
    testset = GrayDataset(img_dir = test_img_dir, mask_dir = test_truth_dir)
    net = get_models(model_name=args.model, is_cls=True, args=args)
    gradcam = GradCam(model=net,gradCamLayer=[net.decoder.layers[-1].conv.double_conv[-1]])

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print(f'Loading model {args.weight}')
    # net.load_state_dict(torch.load(args.weight, map_location="cpu",))
    gradcam.load_weights(args.weight)
    print('Model loaded!')
    # predict one images
    if args.imgdir:
        imgs = os.listdir(args.imgdir)
        imgs = [os.path.join(args.imgdir,i)for i in imgs]
        for i in imgs:
            predict_mask(net=net,imgpath=i,plot_ent=False,outdir=args.outdir)
            # gradcam.plot_cams(i)

    elif args.imgpath:
        predict_mask(net=net,imgpath=args.imgpath,plot_ent=False)
        gradcam.plot_cams(args.imgpath,class_ndx=1)

    # evaluate images
    if args.eval:
        performance = evaluate_imgs(net=net,testdataset=testset)
        print(f'{performance}')

def predict_mask(net,imgpath:str, plot_ent:bool, outdir:str = "./"):
    plotter = Plotter()
    net.to(device="cpu")
    net.eval()
    img = torch.from_numpy(cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE))[None,None,:,:]
    img = img.to(dtype=torch.float32, device='cpu')
    logit, mask_pred = net.targetDomainPredict(img,tsne=False)
    if plot_ent:
        plotter.plot_entropy(torch.softmax(logit, dim=1),saved=True,is_heat=True, imgname=os.path.basename(imgpath).split(".")[0])
    mask_pred = mask_pred.squeeze().numpy()
    mask_pred = mask_pred.astype(np.uint8)*255
    im = Image.fromarray(mask_pred)
    name = os.path.join(outdir,f"predict_{os.path.basename(imgpath)}")
    im.save(name)

    return mask_pred

def evaluate_imgs(net,
                testdataset,):
    # net.eval() #miou 計算不用 eval mode 因為 running mean and running std 誤差可能在訓練過程紀錄的時候過大
    net.mode("eval")
    evaluation_dict = {}
    if not evaluation_dict.get("miou"):
        evaluation_dict["miou"] = []
    if not evaluation_dict.get("dice"):
        evaluation_dict["dice"] = []

    for i,(img, truth) in enumerate(tqdm(testdataset)):
        img = img.unsqueeze(0)#加入批次軸
        img = img.to(dtype=torch.float32)
        truth = truth.unsqueeze(0).to(dtype=torch.int64)#加入批次軸
        #print('shape of truth: ',truth.shape)
        with torch.no_grad():
            logit, mask_pred = net.model.targetDomainPredict(img)   
            # mask_pred = torch.argmax(torch.softmax(logit,dim=1),dim=1,keepdim=True).to(torch.int64) # (1,1,h ,w)

            #compute the mIOU and dice score
            miou = compute_mIoU(mask_pred.numpy(), truth.numpy())
            f1 = dice_score(mask_pred.detach(), truth.detach())
            # print(miou)
            evaluation_dict["miou"].append(miou)
            evaluation_dict["dice"].append(f1)

    for k in evaluation_dict:
        evaluation_dict[k] = sum(evaluation_dict[k]) / len(evaluation_dict[k])

    return evaluation_dict

if __name__ == '__main__':
    main()
