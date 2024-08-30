import os
import yaml
from typing import List
import torch
import torch.nn as nn
from modules.clip import clip
from modules.clip.model import CLIP
from modules.classifier import SimpleClassifier

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load_all(f.read())
        return list(data)[0]

def creat_clip(cfg_dir, name):
    cfg_path = os.path.join(cfg_dir, name + '.yaml')
    if os.path.exists(cfg_path):
        cfg = read_yaml(cfg_path)
        embed_dim = cfg['embed_dim']
        image_resolution = cfg['image_resolution']
        vision_layers = cfg['vision_layers']
        vision_width = cfg['vision_width']
        vision_patch_size = cfg['vision_patch_size']
        context_length = cfg['context_length']
        vocab_size = cfg['vocab_size']
        transformer_width = cfg['transformer_width']
        transformer_heads = cfg['transformer_heads']
        transformer_layers = cfg['transformer_layers']

        model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                     context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
        return model
    else:
        raise RuntimeError(f"Config file {cfg_dir} not found.")


class encoder_text_clip(nn.Module):
    def __init__(self,
                 token_emb,
                 pos_emb,
                 transformer,
                 ln_final,
                 text_proj,
                 dtype):
        super(encoder_text_clip, self).__init__()
        self.token_embedding = token_emb
        self.positional_embedding = pos_emb
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_proj
        self.dtype = dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

def factory_text_enc(opt):
    if opt['name'] == 'clip':
        if os.path.isfile(opt['type']):
            model_path = opt['type']
            model, _ = clip.load(model_path, device='cpu', jit=False)
        elif opt['type'] in _MODELS:
            if opt['pre_trained']:
                model, _ = clip.load(opt['type'], device='cpu', jit=False)
            else:
                model = creat_clip(opt['cfg_dir'], opt['type'])
        else:
            model = None
            raise RuntimeError(f"Model {opt['type']} not found; available models = {available_models()}")

        if model:
            token_emb = model.token_embedding
            pos_emb = model.positional_embedding
            transformer = model.transformer
            ln_final = model.ln_final
            text_proj = model.text_projection
            dtype = model.visual.conv1.weight.dtype
            text_enc = encoder_text_clip(token_emb=token_emb,
                                         pos_emb=pos_emb,
                                         transformer=transformer,
                                         ln_final=ln_final,
                                         text_proj=text_proj,
                                         dtype=dtype)


    else:
        text_enc = None

    return text_enc

def factory_classif(opt):
    input_dim = opt['input_dim']
    m_dim = opt['m_dim']
    num_classes = 2
    dropout = opt['dropout']
    classif = SimpleClassifier(input_dim, m_dim, num_classes, dropout)
    return classif

class TextonlyNet(nn.Module):
    def __init__(self,
                 textual={},
                 classif={}):
        super(TextonlyNet, self).__init__()
        self.textual = factory_text_enc(textual)
        self.classif = factory_classif(classif)
        self.init_parameters()

    def encode_text(self, text):
        return self.textual(text)

    def forward(self, batch):
        text = batch['caption']
        text_feature = self.encode_text(text)
        logits = self.classif(text_feature)
        out = {'logits': logits}
        return out

    def init_parameters(self):
        # for name, param in self.textual.named_parameters():
            # param.requires_grad = False
        for name, param in self.named_parameters():
            param.requires_grad = True


