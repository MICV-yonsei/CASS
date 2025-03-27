import os
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
from PIL import Image
from dinov2.models.vision_transformer import vit_base
import os
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


class Dinov2SelfAttention(nn.Module):
    def __init__(self, arch='vit_base', model_path = None, image_size=(518, 518), patch_size=14, device=None):
        super(Dinov2SelfAttention, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        if arch == 'vit_base':
        # Load the model
            self.model = vit_base(
                patch_size=self.patch_size,
                img_size=self.image_size[0],
                init_values=1.0,
                num_register_tokens=4,
                block_chunks=0
            )
            self.model.load_state_dict(torch.load(model_path))
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)
        self.model.eval()

        # Define the transformation
        self.transform = pth_transforms.Compose([
            pth_transforms.Resize(self.image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def preprocess_image(self, img):
        """Preprocess the image and make it compatible with the model input."""

        img = self.transform(img)
        w, h = img.shape[1] - img.shape[1] % self.patch_size, img.shape[2] - img.shape[2] % self.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        return img

    def forward(self, img):

        img = self.preprocess_image(img)

        """Extract the attention from the model."""
        # Calculate width and height of feature map
        w_featmap = img.shape[-2] // self.patch_size
        h_featmap = img.shape[-1] // self.patch_size

        # Get the last self-attention layer's output
        attentions = self.model.get_last_self_attention(img.to(self.device))

        return attentions 
