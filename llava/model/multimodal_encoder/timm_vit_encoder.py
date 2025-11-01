import os
import torch
import torch.nn as nn
import torchvision.transforms as T

from dataclasses import dataclass

from timm.models.vision_transformer import vit_base_patch16_224


@dataclass
class _TimmViTConfig:
    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 768


class _SimpleImageProcessor:
    def __init__(self, image_size: int = 224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.size = {'shortest_edge': image_size}
        self.crop_size = {'height': image_size, 'width': image_size}
        self.image_mean = mean
        self.image_std = std
        self._to_tensor = T.ToTensor()
        self._resize = T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR)
        self._center_crop = T.CenterCrop(image_size)

    def preprocess(self, image, return_tensors='pt'):
        x = self._to_tensor(image)
        x = self._resize(x)
        x = self._center_crop(x)
        if return_tensors == 'pt':
            return {'pixel_values': x.unsqueeze(0)}
        return {'pixel_values': x}


class TimmVisionTower(nn.Module):
    def __init__(self, delay_load: bool = False):
        super().__init__()
        assert vit_base_patch16_224 is not None, 'timm 未安装，请先安装 timm。'

        self.is_loaded = False

        self._config = _TimmViTConfig()
        self.image_processor = _SimpleImageProcessor(image_size=self._config.image_size)

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('Timm ViT 已加载，跳过重复加载。')
            return

        ckpt_path = os.environ.get('XRCLIP_CKPT', None)

        self.vision_tower = vit_base_patch16_224(in_chans=1)
        self.vision_tower.head = nn.Identity()
        if ckpt_path is None:
            raise FileNotFoundError('环境变量 XRCLIP_CKPT 未设置，无法加载权重。请 `export XRCLIP_CKPT=/abs/path/to/xr_clip_ckpt.pth`')
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.vision_tower.load_state_dict(state, strict=True)

        print("Loaded XRCLIP weights")

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        x = images.to(device=self.device, dtype=self.dtype)
        if x.shape[1] == 3:
            x = x[:, :1]

        tokens = None
        if hasattr(self.vision_tower, 'forward_features'):
            try:
                tokens = self.vision_tower.forward_features(x, return_all_tokens=True)
            except TypeError:
                tokens = self.vision_tower.forward_features(x)
        if tokens is None:
            tokens = self.vision_tower(x)

        if tokens.dim() == 4:
            B, C, H, W = tokens.shape
            tokens = tokens.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif tokens.dim() == 3:
            if tokens.size(1) > 1:
                tokens = tokens[:, 1:, :]
        else:
            raise RuntimeError('ViT 输出形状不符合预期')

        return tokens.to(images.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self._config

    @property
    def hidden_size(self):
        return self._config.hidden_size

    @property
    def num_patches_per_side(self):
        return self._config.image_size // self._config.patch_size

    @property
    def num_patches(self):
        return (self._config.image_size // self._config.patch_size) ** 2

