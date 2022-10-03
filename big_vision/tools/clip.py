"""Drop-in replacement for CLIP when using LiT models.

Note: models/tokenizers are stored in private bucket. To access them make sure
to first execute `gcloud auth application-default login` in a shell.
"""

import os
from typing import List, Union

from absl import logging
from big_vision.pp import ops_text
from big_vision.tools import torch as bv_torch
import numpy as np
import packaging
import PIL
from tensorflow.io import gfile
import torch
from torch import nn
import torchvision

_tokenizers = {}

# download paths & configs
_MODELS = {
    'lit_jft_b32b': (
        'gs://big_vision_eu/lit_private/lit_jft_b32b/checkpoint.npz',
        dict(
            image=dict(variant='B/32', pool_type='map_buggy'),
            text=dict(variant='B'),
            out_dim=[0, 768],
        ),
    ),
    'lit_jft_b16b': (
        'gs://big_vision_eu/lit_private/lit_jft_b16b/checkpoint.npz',
        dict(
            image=dict(variant='B/16', pool_type='map_buggy'),
            text=dict(variant='B'),
            out_dim=[0, 768],
        ),
    ),
    'lit_jft_l16l': (
        'gs://big_vision_eu/lit_private/lit_jft_l16l/checkpoint.npz',
        dict(
            image=dict(variant='L/16', pool_type='map_buggy'),
            text=dict(variant='L'),
            out_dim=[0, 1024],
        ),
    ),
    'lit_jft_g14g': (
        'gs://big_vision_eu/lit_private/lit_jft_g14g/checkpoint.npz',
        dict(
            image=dict(variant='g/14', pool_type='map'),
            text=dict(variant='g'),
            out_dim=[0, 1408],
        ),
    ),
    'lit_jft_e14g': (
        'gs://big_vision_eu/lit_private/lit_jft_e14g/checkpoint.npz',
        dict(
            image=dict(variant='e/14', pool_type='map'),
            text=dict(variant='g'),
            out_dim=[0, 1792],
        ),
    ),
}

_TOKENIZER_MODELS = {
  'lit_jft_e14g': 'gs://big_vision_eu/lit_private/lit_jft_e14g//argus_dedup_v1.2_288M32k.model',
}


class LiT(nn.Module):
  """CLIP-like wrapper for LiT models."""

  def __init__(self, lit_model):
    super().__init__()
    self.lit_model = lit_model
    self.visual = lit_model.img

  def initialize_parameters(self):
    raise NotImplementedError
  def build_attention_mask(self):
    raise NotImplementedError

  @property
  def dtype(self):
    return self.lit_model.img.embedding.weight.dtype

  def encode_image(self, image):
    return self.visual(image.type(self.dtype))

  def encode_text(self, text):
    return self.lit_model.txt(text)

  def load_weights(self, weights):
    self.lit_model.load(weights)

  def forward(self, image, text):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.lit_model.t.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


class _LoggingDict(dict):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.accessed = {}

  def __getitem__(self, k):
    self.accessed[k] = self.accessed.get(k, 0) + 1
    return super().__getitem__(k)


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):

  download_root = download_root or os.path.expanduser("~/.cache/lit")
  os.makedirs(download_root, exist_ok=True)
  url, config = _MODELS[name]
  path = os.path.join(download_root, f'{name}.npz')
  if not os.path.exists(path):
    logging.info('Downloading %s to %s', url, path)
    gfile.copy(url, path)

  logging.info('Loading model from %s (%.1f MB)', path, os.stat(path).st_size / 1e6)
  model = bv_torch.TwoTowers(**config)
  with open(path, 'rb') as f:
    w = _LoggingDict(**np.load(path))
  model.load(w, prefix='params/' if next(iter(w)).startswith('params/') else '')
  unused = set(w).difference(w.accessed)
  assert not unused, unused
  overused = {k: v for k, v in w.accessed.items() if v > 1}
  assert not overused, overused

  # https://github.com/openai/CLIP/blob/main/clip/clip.py
  device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
  device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]
  def patch_device(module):
    try:
      graphs = [module.graph] if hasattr(module, "graph") else []
    except RuntimeError:
      graphs = []

    if hasattr(module, "forward1"):
      graphs.append(module.forward1.graph)

    for graph in graphs:
      for node in graph.findAllNodes("prim::Constant"):
        if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
          node.copyAttributes(device_node)

  model.apply(patch_device)
  patch_device(model.img)
  patch_device(model.txt)

  pp_img = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Resize([224, 224], interpolation=PIL.Image.BILINEAR),
      torchvision.transforms.Normalize(mean=0.5, std=0.5),
  ])

  return LiT(model), pp_img


def tokenize(texts: Union[str, List[str]], context_length: int = 16, truncate: bool = True, name: str = '') -> Union[torch.IntTensor, torch.LongTensor]:
  if not truncate:
    warnings.warn('Ignoring truncate=False!')
  if isinstance(texts, str):
      texts = [texts]
  global _tokenize
  if name not in _TOKENIZER_MODELS:
    name = None
  if name not in _tokenizers:
    kw = {}
    if name in _TOKENIZER_MODELS:
      kw['model'] = _TOKENIZER_MODELS[name]
    _tokenizers[name] = ops_text.get_pp_tokenize(
        max_len=context_length,
        eos='sticky',
        inkey='text',
        outkey='tokens',
        pad_value=1,
        **kw,
    )
  tokenizer = _tokenizers[name]
  
  dtype = torch.int
  if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
    dtype = torch.long
  result = torch.zeros(len(texts), context_length, dtype=dtype)
  for i, text in enumerate(texts):
    result[i, :] = torch.tensor(tokenizer({'text': text})['tokens'].numpy(), dtype=dtype)

  return result
