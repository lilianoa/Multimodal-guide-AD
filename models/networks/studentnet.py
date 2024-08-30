import os
import torch
import torch.nn as nn
from modules.clip import clip
from modules.clip.model import ModifiedResNet, VisionTransformer
from typing import List
from modules.classifier import SimpleClassifier

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

def build_visual_clip(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["textual.text_projection"].shape[1]
    if isinstance(vision_layers, (tuple, list)):
        vision_heads = vision_width * 32 // 64
        visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width
        )
    else:
        vision_heads = vision_width // 64
        visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
        visual.load_state_dict(state_dict['visual'])
    return visual


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def factory_img_enc(opt):
    if opt['name'] == 'clip':
        if os.path.isfile(opt['type']):
            model_path = opt['type']
            state_dict = torch.load(model_path)
            if type(state_dict) is dict and 'network' in state_dict.keys():  # from checkpoint
                t_network = state_dict['network']
                img_enc = build_visual_clip(t_network)
            else:
                model, _ = clip.load(model_path, device='cpu', jit=False)
                img_enc = model.visual
        elif opt['type'] in _MODELS:
            model, _ = clip.load(opt['type'], device='cpu', jit=False)
            img_enc = model.visual
        else:
            raise RuntimeError(f"Model {opt['type']} not found; available models = {available_models()}")
            img_enc = None

    else:
        img_enc = None

    return img_enc

def factory_classif(opt):
    input_dim = opt['input_dim']
    m_dim = opt['m_dim']
    num_classes = 2
    dropout = opt['dropout']
    classif = SimpleClassifier(input_dim, m_dim, num_classes, dropout)
    return classif


class StudentNet(nn.Module):
    def __init__(self,
                 visual={},
                 classif={}):
        super(StudentNet, self).__init__()
        self.visual = factory_img_enc(visual)
        self.classif = factory_classif(classif)
        self.init_parameters()

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def forward(self, batch):  # image
        image = batch['img']
        image_features = self.encode_image(image)
        logits = self.classif(image_features)
        out = {'logits': logits}
        return out

    def init_parameters(self):
        # for name, param in self.visual.named_parameters():
            # param.requires_grad = False
        for name, param in self.named_parameters():
            param.requires_grad = True

