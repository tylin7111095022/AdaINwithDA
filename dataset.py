import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CSDataset(Dataset):
    f'''
    回傳img(torch.uint8)及label(torch.float32)\n
    這個dataset 最後回傳的圖片為RGB圖
    '''
    def __init__(self,content_dir,style_dir,transform = None):
        assert len(os.listdir(content_dir)) == len(os.listdir(style_dir)), "numbers of img and label dismatch."
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.img_list = [i for i in os.listdir(content_dir)]
        self.transforms = transform

    def __getitem__(self, idx):
        # content image
        img_name = self.img_list[idx]
        img_a = cv2.imread(os.path.join(self.content_dir, img_name), cv2.IMREAD_GRAYSCALE)
        if img_a.ndim == 2:
            img_a = np.expand_dims(img_a,2) #如果沒通道軸，加入通道軸
        img = torch.permute(torch.from_numpy(img_a),(2,0,1))
        img = img.to(torch.float32) / 255 #[0, 255] -> [0, 1]
        imgsize = (img.shape[1], img.shape[2])
        # style image
        label_name = img_name
        label_path = os.path.join(self.style_dir, label_name)
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        if label.ndim == 2:
            label = np.expand_dims(label,2) #如果沒通道軸，加入通道軸 為了執行後面的cv2.threshold
        label = cv2.resize(label, (imgsize[1],imgsize[0]), cv2.INTER_NEAREST) #將label resize成跟 img  一樣的長寬
        label_t = torch.from_numpy(label).unsqueeze(2).to(torch.float32)
        label_t = torch.permute(label_t,(2,0,1))
        label_t = label_t.to(torch.float32) / 255 #[0, 255] -> [0, 1]

        if self.transforms:
            img = self.transform(img)
            label_t = self.transform(label_t)

        return img,label_t

    def __len__(self):
        return len(self.img_list)

class GrayDataset(Dataset):
    f'''
    回傳img(torch.uint8)及label(torch.float32)\n
    這個dataset 最後回傳的圖片為灰階圖
    '''
    def __init__(self,img_dir,mask_dir,transform = None):
        assert len(os.listdir(img_dir)) == len(os.listdir(mask_dir)), "numbers of img and label dismatch."
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = [i for i in os.listdir(img_dir)]
        
        self.transforms = transform

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_a = cv2.imread(os.path.join(self.img_dir, img_name),cv2.IMREAD_GRAYSCALE)
        if img_a.ndim == 2:
            img_a = np.expand_dims(img_a,2) #如果沒通道軸，加入通道軸
        img = torch.permute(torch.from_numpy(img_a),(2,0,1))
        img = img.to(torch.float32) / 255 #[0, 255] -> [0, 1]
        imgsize = (img.shape[1], img.shape[2])

        # label
        label_name = img_name
        label_path = os.path.join(self.mask_dir, label_name)
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        if label.ndim == 2:
            label = np.expand_dims(label,2) #如果沒通道軸，加入通道軸
        label = cv2.resize(label, (imgsize[1],imgsize[0]), cv2.INTER_NEAREST) #將label resize成跟 img  一樣的長寬
        #label內的值不只兩個，這導致除以255後值介於0~1的值在後續計算iou將label轉回int64的時候某些值被無條件捨去成0
        ret,label_binary = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
        #print(np.unique(label_binary))
        label_t = torch.from_numpy(label_binary).unsqueeze(0).to(torch.int64)#加入通道軸
        # 處理標籤，将像素值255改為1
        if label_t.max() > 1:
            label_t[label_t == 255] = 1

        if self.transforms:
            img = self.transform(img)
            label_t = self.transform(label_t)

        return img,label_t

    def __len__(self):
        return len(self.img_list)
    
class AdainDataset(Dataset):
    
    def __init__(self,content_dir,truth_dir,style_dir,transform = None):
        assert len(os.listdir(content_dir)) == len(os.listdir(style_dir)), "numbers of img and label dismatch."
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.truth_dir = truth_dir
        self.img_list = [i for i in os.listdir(content_dir)]
        self.transforms = transform

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_a = cv2.imread(os.path.join(self.content_dir, img_name), cv2.IMREAD_GRAYSCALE)
        if img_a.ndim == 2:
            img_a = np.expand_dims(img_a,2) #如果沒通道軸，加入通道軸
        img = torch.permute(torch.from_numpy(img_a),(2,0,1))
        img = img.to(torch.float32) / 255 #[0, 255] -> [0, 1]
        imgsize = (img.shape[1], img.shape[2])
        # style image
        style_path = os.path.join(self.style_dir, img_name)
        style = cv2.imread(style_path,cv2.IMREAD_GRAYSCALE)
        if style.ndim == 2:
            style = np.expand_dims(style,2) #如果沒通道軸，加入通道軸 為了執行後面的cv2.threshold
        style = cv2.resize(style, (imgsize[1],imgsize[0]), cv2.INTER_NEAREST) #將style resize成跟 img  一樣的長寬
        style_t = torch.from_numpy(style).unsqueeze(2).to(torch.float32)
        style_t = torch.permute(style_t,(2,0,1))
        style_t = style_t.to(torch.float32) / 255 #[0, 255] -> [0, 1]
        # label image
        label_path = os.path.join(self.truth_dir, img_name)
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        if label.ndim == 2:
            label = np.expand_dims(label,2) #如果沒通道軸，加入通道軸
        label = cv2.resize(label, (imgsize[1],imgsize[0]), cv2.INTER_NEAREST) #將label resize成跟 img  一樣的長寬
        #label內的值不只兩個，這導致除以255後值介於0~1的值在後續計算iou將label轉回int64的時候某些值被無條件捨去成0
        ret,label_binary = cv2.threshold(label,127,255,cv2.THRESH_BINARY)
        #print(np.unique(label_binary))
        label_t = torch.from_numpy(label_binary).unsqueeze(0).to(torch.int64)#加入通道軸
        # 處理標籤，将像素值255改為1
        if label_t.max() > 1:
            label_t[label_t == 255] = 1

        if self.transforms:
            img = self.transform(img)
            label_t = self.transform(label_t)
            style_t = self.transform(style_t)

        return img,label_t,style_t
    
    def __len__(self):
        return len(self.img_list)
    
if __name__ == "__main__":
    # ds = STDataset(student_dir = r'data\real_B', teacher_dir= r'data\fake_A')
    ds = AdainDataset(content_dir = r"data\real_A", truth_dir=r"data\train_mask",style_dir=r"data\fake_B")
    img, mask, style = ds[3]
    print(img.shape)
    print(mask.shape)
    print(style.shape)
    print(np.unique(img.permute(1,2,0).numpy()))
    print(np.unique(mask.permute(1,2,0).numpy()))
    print(np.unique(style.permute(1,2,0).numpy()))

    cv2.imwrite("img.jpg", 255*img.permute(1,2,0).numpy())
    cv2.imwrite("mask.jpg", 255*(mask).permute(1,2,0).numpy(),)
    cv2.imwrite("style.jpg", 255*(style).permute(1,2,0).numpy(),)