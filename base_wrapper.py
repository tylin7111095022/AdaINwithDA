import torch.nn as nn
import typing as Optional
import torch
import numpy as np
import cv2
from utils import ActivationandGradients

class ModelWrapper(object):
    def __init__(self, model:nn.Module, gradCamLayer:Optional[list]=None):
        self.model = model
        self.targetLayers = gradCamLayer
        self.activationAndGrads = ActivationandGradients(model,gradCamLayer)
        self.device = next(self.model.parameters()).device
        self.optim = None
        self.loss_fn = None

    def __call__(self, x):
        logits = self.activationAndGrads(x)
        return logits

    # def forward(self, x):
    #     logits = self.activationAndGrads(x)
    #     return logits

    def Init_optim(self, lr):
        self.optim = torch.optim.Adam(self.model.parameters(),lr = lr,betas=(0.9,0.999))

    def Init_loss(self):
        pass

    def train_one_batch(self, x, truth):
        pass

    def mode(self,mode:str):
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            print("Error, mode should be train or eval !!!, mode is eval now.")
            self.model.eval()
    
    def load_weights(self, weights):
        model_dict = self.model.state_dict()
        pretrained_w_dict = torch.load(weights)
        weights_dict = {k:v for k, v in pretrained_w_dict.items() if k in model_dict }
        self.model.load_state_dict(weights_dict)

    def to(self, device:str):
        self.model.to(device=device)
        self.device = next(self.model.parameters()).device
    
    def get_cam_images(self,x):
        # logits = self.forward(x)
        logits = self(x)
        target_size = tuple(*(logits.shape[-2:]))
        cam_per_target_layer = []
        weights = []
        cams = []
        # get cam weights
        for grad in self.activationAndGrads.gradients:
            weight = np.mean(grad, axis=[2,3])
            weights.append(weight)
        # get cam activations
        activations = self.activationAndGrads.activations
        for w, a in zip(weights, activations):
            weighted_activations = w[:, :, None, None] * a
            cam = torch.sum(weighted_activations,dim=1)
            cam = torch.where(cam > 0., cam, 0.)  # relu, cam shape: [b,h,w]
            cams.append(cam)
        for cam in cams:
            scaled_cam = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled_cam[:,None]) # insert channel axis shape: [b,1,h,w]

        return cam_per_target_layer

    def scale_cam_image(self, cams, target_size=None):
        result = []
        for img in cams: # img shape [h,w]
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img)) # normalize
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.stack(result,axis=0)
        result = np.float32(result) # result shape [b,h,w]

        return result