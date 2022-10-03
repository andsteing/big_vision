"""Re-implementation of some selected models in PyTorch.

Class names, configuration options, and module names exactly mirror the names
from the corresponding JAX implementations in the models/ directory.
"""

import torch
from torch import nn


# from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
def _n2p(w, t=True):
  if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
    w = w.flatten()
  if t:
    if w.ndim == 4:
      w = w.transpose([3, 2, 0, 1])
    elif w.ndim == 3:
      w = w.transpose([2, 0, 1])
    elif w.ndim == 2:
      w = w.transpose([1, 0])
  return torch.from_numpy(w)


class MlpBlock(nn.Module):
  """From big_vision.models.vit"""

  def __init__(self, dim=768, *, mlp_dim=0):
    super().__init__()
    if not mlp_dim: mlp_dim = 4 * dim
    self.Dense_0 = nn.Linear(dim, mlp_dim)
    self.act = nn.GELU()
    self.Dense_1 = nn.Linear(mlp_dim, dim)

  def forward(self, x):
    x = self.Dense_0(x)
    x = self.act(x)
    x = self.Dense_1(x)
    return x

  @torch.no_grad()
  def load(self, w, prefix=''):
    for r in range(2):
      getattr(self, f'Dense_{r}').weight.copy_(_n2p(w[f'{prefix}Dense_{r}/kernel']))
      getattr(self, f'Dense_{r}').bias.copy_(_n2p(w[f'{prefix}Dense_{r}/bias']))


class MultiHeadDotProductAttention(nn.Module):
  """From flax.linen"""

  def __init__(self, dim=768, *, num_heads=12):
    super().__init__()
    assert dim % num_heads == 0, 'dim should be divisible by num_heads'
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim**-0.5

    self.key = nn.Linear(dim, dim)
    self.query = nn.Linear(dim, dim)
    self.value = nn.Linear(dim, dim)
    self.out = nn.Linear(dim, dim)

  def forward(self, x_q, x_kv=None):
    if x_kv is None:
      x_kv = x_q
    B, N, C = x_q.shape
    B2, N2, C2 = x_kv.shape
    assert B == B2 and C == C2
    c = C // self.num_heads
    q = self.query(x_q).reshape(B, N, self.num_heads, c).permute(0, 2, 1, 3)
    k = self.key(x_kv).reshape(B, N2, self.num_heads, c).permute(0, 2, 1, 3)
    v = self.value(x_kv).reshape(B, N2, self.num_heads, c).permute(0, 2, 1, 3)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.out(x)
    return x

  @torch.no_grad()
  def load(self, w, prefix=''):
    for kqv in ('key', 'query', 'value'):
      getattr(self, kqv).weight.copy_(_n2p(w[f'{prefix}{kqv}/kernel'], t=False).flatten(1).T)
      getattr(self, kqv).bias.copy_(_n2p(w[f'{prefix}{kqv}/bias'], t=False).flatten().T)
    self.out.weight.copy_(_n2p(w[f'{prefix}out/kernel']).flatten(1))
    self.out.bias.copy_(_n2p(w[f'{prefix}out/bias']).flatten())


class Encoder1DBlock(nn.Module):
  """From big_vision.models.vit"""

  def __init__(self, dim=768, *, num_heads=12, mlp_dim=0):
    super().__init__()
    self.LayerNorm_0 = nn.LayerNorm(dim)
    self.MultiHeadDotProductAttention_0 = MultiHeadDotProductAttention(dim, num_heads=num_heads)
    self.LayerNorm_1 = nn.LayerNorm(dim)
    self.MlpBlock_0 = MlpBlock(dim, mlp_dim=mlp_dim)

  def forward(self, x):
    y = self.LayerNorm_0(x)
    y = self.MultiHeadDotProductAttention_0(y)
    x = x + y
    y = self.LayerNorm_1(x)
    y = self.MlpBlock_0(y)
    return x + y

  @torch.no_grad()
  def load(self, w, prefix=''):
    for i in range(2):
      getattr(self, f'LayerNorm_{i}').weight.copy_(_n2p(w[f'{prefix}LayerNorm_{i}/scale']))
      getattr(self, f'LayerNorm_{i}').bias.copy_(_n2p(w[f'{prefix}LayerNorm_{i}/bias']))
    self.MultiHeadDotProductAttention_0.load(w, prefix=f'{prefix}MultiHeadDotProductAttention_0/')
    self.MlpBlock_0.load(w, prefix=f'{prefix}MlpBlock_0/')


class Encoder(nn.Module):
  """From big_vision.models.vit"""

  def __init__(self, depth, *, dim=768, num_heads=12, mlp_dim=0):
    super().__init__()
    self.depth = depth
    for i in range(self.depth):
      setattr(self, f'encoderblock_{i}', Encoder1DBlock(dim, num_heads=num_heads, mlp_dim=mlp_dim))
    self.encoder_norm = nn.LayerNorm(dim)

  def forward(self, x):
    for i in range(self.depth):
      x = getattr(self, f'encoderblock_{i}')(x)
    return self.encoder_norm(x)

  @torch.no_grad()
  def load(self, w, prefix=''):
    for i in range(self.depth):
      getattr(self, f'encoderblock_{i}').load(w, prefix=f'{prefix}encoderblock_{i}/')
    self.encoder_norm.weight.copy_(_n2p(w[f'{prefix}encoder_norm/scale']))
    self.encoder_norm.bias.copy_(_n2p(w[f'{prefix}encoder_norm/bias']))


class MAPHead(nn.Module):
  """From big_vision.models.vit"""

  def __init__(self, dim=768, *, mlp_dim=0, num_heads=12, buggy=False):
    super().__init__()
    self.buggy = buggy
    self.probe = nn.Parameter(torch.zeros([1, 1, dim]))
    self.MultiHeadDotProductAttention_0 = MultiHeadDotProductAttention(dim, num_heads=num_heads)
    self.MlpBlock_0 = MlpBlock(dim, mlp_dim=mlp_dim)
    self.LayerNorm_0 = nn.LayerNorm(dim, eps=1e-6)

  def forward(self, x):
    B, N, C = x.shape
    probe = torch.cat([self.probe for _ in range(B)])
    x = self.MultiHeadDotProductAttention_0(probe, x)
    y = self.LayerNorm_0(x)
    if self.buggy:
      x = y
    x = x + self.MlpBlock_0(y)
    return x[:, 0]

  @torch.no_grad()
  def load(self, w, prefix=''):
    self.probe.copy_(_n2p(w[f'{prefix}probe'], t=False))
    self.MultiHeadDotProductAttention_0.load(w, prefix=f'{prefix}MultiHeadDotProductAttention_0/')
    self.MlpBlock_0.load(w, prefix=f'{prefix}MlpBlock_0/')
    self.LayerNorm_0.weight.copy_(_n2p(w[f'{prefix}LayerNorm_0/scale']))
    self.LayerNorm_0.bias.copy_(_n2p(w[f'{prefix}LayerNorm_0/bias']))


class ViT(nn.Module):
  """From big_vision.models.vit"""

  def __init__(self, num_classes, *, patch_size=16, width=768, depth=12, num_heads=12, mlp_dim=0, pool_type='gap', img_size=224):
    super().__init__()
    self.embedding = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
    assert img_size % patch_size == 0, (img_size, patch_size)
    grid_size = img_size // patch_size
    self.pos_embedding = nn.Parameter(torch.zeros(1, grid_size * grid_size, width))
    self.Transformer = Encoder(depth, dim=width, num_heads=num_heads, mlp_dim=mlp_dim)
    self.pool_type = pool_type
    if pool_type in ('map', 'map_buggy'):
      self.MAPHead_0 = MAPHead(width, mlp_dim=mlp_dim, num_heads=num_heads, buggy=pool_type == 'map_buggy')
    elif pool_type != 'gap':
      raise ValueError(pool_type)
    if num_classes:
      self.head = nn.Linear(width, num_classes)

  def forward(self, x):
    x = self.embedding(x).flatten(2).transpose(1, 2)
    x += self.pos_embedding
    x = self.Transformer(x)
    if hasattr(self, 'MAPHead_0'):
      x = self.MAPHead_0(x)
    if hasattr(self, 'head'):
      x = self.head(x)
    if self.pool_type == 'gap':
      x = x.mean(axis=1)
    return x

  @torch.no_grad()
  def load(self, w, prefix=''):
    self.embedding.weight.copy_(_n2p(w[f'{prefix}embedding/kernel']))
    self.embedding.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    self.pos_embedding.copy_(_n2p(w[f'{prefix}pos_embedding'], t=False))
    self.Transformer.load(w, prefix=f'{prefix}Transformer/')
    if hasattr(self, 'MAPHead_0'):
      self.MAPHead_0.load(w, prefix=f'{prefix}MAPHead_0/')
    if hasattr(self, 'head'):
      self.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
      self.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))


class TextTransformer(nn.Module):
  """From big_vision.models.proj.image_text.text_transformer"""

  def __init__(self, num_classes, *, width=512, depth=12, num_heads=8, mlp_dim=2048, pool_type='last', max_len=16, vocab_size=32_000):
    super().__init__()
    self.Embed_0 = nn.Embedding(vocab_size, width)
    self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, width))
    self.Encoder_0 = Encoder(depth, dim=width, num_heads=num_heads, mlp_dim=mlp_dim)
    self.pool_type = pool_type
    if pool_type != 'last':
      raise ValueError(pool_type)
    if num_classes:
      self.head = nn.Linear(width, num_classes)

  def forward(self, x):
    x = self.Embed_0(x)
    x += self.pos_embedding
    x = self.Encoder_0(x)
    if hasattr(self, 'MAPHead_0'):
      x = self.MAPHead_0(x)
    if self.pool_type == 'last':
      x = x[:, -1]
    if hasattr(self, 'head'):
      x = self.head(x)
    return x

  @torch.no_grad()
  def load(self, w, prefix=''):
    self.Embed_0.weight.copy_(_n2p(w[f'{prefix}Embed_0/embedding'], t=False))
    self.pos_embedding.copy_(_n2p(w[f'{prefix}pos_embedding'], t=False))
    self.Encoder_0.load(w, prefix=f'{prefix}Encoder_0/')
    if hasattr(self, 'head'):
      self.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
      self.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))


_IMAGE_VARIANTS = {
    'B/32': dict(
        patch_size=32,
    ),
    'B/16': dict(),
    'L/16': dict(
        width=1024,
        depth=24,
        num_heads=16,
    ),
    'L/16': dict(
        width=1024,
        depth=24,
        num_heads=16,
    ),
    'g/14': dict(
        patch_size=14,
        img_size=280,
        width=1408,
        depth=40,
        mlp_dim=6144,
        num_heads=16,
    ),
    'e/14': dict(
        patch_size=14,
        img_size=280,
        width=1792,
        depth=56,
        mlp_dim=15360,
        num_heads=16,
    ),
}
_TEXT_VARIANTS = {
    'B': dict(
        width=768,
        num_heads=12,
        mlp_dim=0,
    ),
    'L': dict(
        width=1024,
        depth=24,
        num_heads=16,
        mlp_dim=0,
    ),
    'g': dict(
        width=1408,
        depth=40,
        mlp_dim=6144,
        num_heads=16,
    ),
}


class TwoTowers(nn.Module):
  """From big_vision.models.proj.image_text.two_towers"""

  def __init__(self, *, image, text, out_dim):
    super().__init__()
    image = {**image}
    variant = image.pop('variant', None)
    if variant:
      image = {**_IMAGE_VARIANTS[variant], **image}
    self.img = ViT(out_dim[0], **image)
    text = {**text}
    variant = text.pop('variant', None)
    if variant:
      text = {**_TEXT_VARIANTS[variant], **text}
    self.txt = TextTransformer(out_dim[1], **text)
    self.t = nn.Parameter(torch.zeros(1))

  def forward(self, image, text=None):
    zimg = ztxt = None
    if image is not None:
      zimg = self.img(image)
      zimg /= zimg.norm(dim=1, keepdim=True) + 1e-8
    if text is not None:
      ztxt = self.txt(text)
      ztxt /= ztxt.norm(dim=1, keepdim=True) + 1e-8
    return zimg, ztxt

  @torch.no_grad()
  def load(self, w, prefix=''):
    self.img.load(w, prefix=f'{prefix}img/')
    self.txt.load(w, prefix=f'{prefix}txt/')
    self.t.copy_(_n2p(w[f'{prefix}t']))
