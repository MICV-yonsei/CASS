import os
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import requests
from io import BytesIO
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import vision_transformer as vits
import warnings
warnings.filterwarnings('ignore')



class DinoSelfAttention(nn.Module):
    def __init__(self, arch='vit_base', patch_size=8, pretrained_weights='', checkpoint_key='teacher',
                 image_size=(224, 224), output_dir='.', threshold=None):
        super(DinoSelfAttention, self).__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.pretrained_weights = pretrained_weights
        self.checkpoint_key = checkpoint_key
        self.image_size = image_size
        self.output_dir = output_dir
        self.threshold = threshold
        
        # Set up device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Build model
        self.model = vits.__dict__[self.arch](patch_size=self.patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(self.device)
        
        # Load pretrained weights
        self._load_weights()

    def _load_weights(self):
        if os.path.isfile(self.pretrained_weights):
            state_dict = torch.load(self.pretrained_weights, map_location="cpu")
            if self.checkpoint_key is not None and self.checkpoint_key in state_dict:
                print(f"Take key {self.checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[self.checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f'Pretrained weights found at {self.pretrained_weights} and loaded with msg: {msg}')
        else:
            print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
            url = self._get_pretrained_url()
            if url is not None:
                print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
                state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
                self.model.load_state_dict(state_dict, strict=True)
            else:
                print("There is no reference weights available for this model => We use random weights.")

    def _get_pretrained_url(self):
        if self.arch == "vit_small" and self.patch_size == 16:
            return "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif self.arch == "vit_small" and self.patch_size == 8:
            return "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif self.arch == "vit_base" and self.patch_size == 16:
            return "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif self.arch == "vit_base" and self.patch_size == 8:
            return "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        return None

    def forward(self, image):
        
        with torch.no_grad():

            transform = pth_transforms.Compose([
                # pth_transforms.Resize(self.image_size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            img = transform(image)
            
            w, h = img.shape[1] - img.shape[1] % self.patch_size, img.shape[2] - img.shape[2] % self.patch_size
            img = img[:, :w, :h].unsqueeze(0)
            
            w_featmap = img.shape[-2] // self.patch_size
            h_featmap = img.shape[-1] // self.patch_size
            
            feat_all, attn_all, qkv_all = self.model.get_intermediate_feat(img.to(self.device))
            feat, attn, qkv = feat_all[-1], attn_all[-1], qkv_all[-1]
            
            return qkv


# Example usage:
if __name__ == '__main__':
    visualizer = DinoSelfAttention(arch='vit_base', patch_size=8,image_size=(224, 224))
    image = Image.open("")
    attention_map = visualizer(image)
