# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wraps `big_vision` PaliGemma model for easy use in Colab etc.

Synopsis::

  import PIL.Image
  img = PIL.Image.open('img.jpg') # !wget -O img.jpg https://picsum.photos/800
  from big_vision.wrappers.proj.paligemma import paligemma_model as model_lib
  config = model_lib.get_config('./pt_2b_224.b16.npz')

  model, params_cpu = model_lib.load_model(config)
  params = model.shard_params(params_cpu)

  batch = model.prepare_batch([img], ['caption en'])
  batch = model.shard_batch(batch)
  tokens = model.generate(params, batch)
  texts = model.tokenizer.to_str(tokens)
  print(texts)

  batch2 = model.prepare_batch(
      [img] * len(texts), ['caption en'] * len(texts), texts)
  batch2 = model.shard_batch(batch2)
  scores = model.score(params, batch2)
  print(scores)
"""

from collections.abc import Callable
import dataclasses
import enum
import functools
import importlib
import json
import re
from typing import Any
import urllib.parse

from big_vision import sharding
from big_vision import utils
from big_vision.models.proj.paligemma import paligemma
import big_vision.models.proj.paligemma.gemma_bv  # pylint: disable=unused-import
import big_vision.models.vit  # pylint: disable=unused-import
from big_vision.pp import builder as pp_builder
from big_vision.pp import ops_general  # pylint: disable=unused-import
from big_vision.pp import ops_image  # pylint: disable=unused-import
from big_vision.pp import ops_text  # pylint: disable=unused-import
from big_vision.pp import tokenizer
from big_vision.pp.proj.paligemma import ops as ops_paligemma  # pylint: disable=unused-import
from big_vision.trainers.proj.paligemma import predict_fns

import flax.linen as nn
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import ml_collections
import numpy as np
import PIL.Image
import tensorflow as tf



DUMMY_MODEL = 'dummy_224'

_PATCH_SIZE = 14


class ShardingStrategy(enum.Enum):
  """Sharding strategy, see (internal link)."""

  FSDP = 'fsdp'
  MEGATRON = 'megatron'


@functools.cache
def _get_devices():
  return mesh_utils.create_device_mesh([len(jax.devices())])


@functools.cache
def _get_mesh():
  return jax.sharding.Mesh(_get_devices(), 'devices')


def _get_sharding_rules():
  return [('act_batch', ('devices',))]


def _init_dummy(path, tokenizer_spec, vocab_size, llm_variant):
  """Returns a small dummy model that can be run on CPU."""
  del llm_variant
  assert path == DUMMY_MODEL, path
  tok = tokenizer.get_tokenizer(tokenizer_spec)
  config = ml_collections.FrozenConfigDict(dict(
      llm_model='proj.paligemma.gemma_bv',
      llm=dict(vocab_size=vocab_size, variant='gemma_test'),
      img=dict(variant='mu/32', pool_type='none', scan=True),
  ))
  model = paligemma.Model(**config)

  def init(rng):
    return model.init(
        rng,
        image=jnp.ones((1, 224, 224, 3)),
        text=jnp.ones((1, 128), dtype=jnp.int32),
        mask_ar=jnp.ones((1, 128), dtype=jnp.int32),
    )
  params_cpu = utils.jit_cpu()(init)(
      jax.device_put(jax.random.PRNGKey(utils.put_cpu(0))),
  )['params']
  return model, params_cpu, predict_fns.get_all(model), tok


def _load(path, tokenizer_spec, vocab_size, llm_variant):
  """Loads model, params, decode functions and tokenizer."""
  tok = tokenizer.get_tokenizer(tokenizer_spec)

  config = ml_collections.FrozenConfigDict(dict(
      llm_model='proj.paligemma.gemma_bv',
      llm=dict(vocab_size=vocab_size, variant=llm_variant),
      img=dict(variant=f'So400m/{_PATCH_SIZE}', pool_type='none', scan=True),
  ))
  model = paligemma.Model(**config)

  params_cpu = paligemma.load(None, path, config)

  return model, params_cpu, predict_fns.get_all(model), tok


def _shard_params(params_cpu, sharding_strategy):
  """Shards `params_cpu` according to `sharding` strategy."""
  if sharding_strategy == ShardingStrategy.FSDP:
    params_sharding = sharding.infer_sharding(
        params_cpu, strategy=[('.*', 'fsdp(axis="devices")')], mesh=_get_mesh()
    )
  elif sharding_strategy == ShardingStrategy.MEGATRON:
    params_sharding = sharding.infer_sharding(
        params_cpu,
        strategy=[
            # pylint: disable=line-too-long
            # pyformat: disable
            (r'img/Transformer/encoderblock/MlpBlock_\d+/Dense_0/kernel', 'shard_dim("devices", -1)'),
            (r'img/Transformer/encoderblock/MlpBlock_\d+/Dense_1/kernel', 'shard_dim("devices", -2)'),
            (r'img/Transformer/encoderblock/MultiHeadDotProductAttention_\d+/(query|key|value)/kernel', 'shard_dim("devices", -2)'),
            (r'img/Transformer/encoderblock/MultiHeadDotProductAttention_\d+/out/kernel', 'shard_dim("devices", -3)'),
            (r'llm/layers/mlp/gating_einsum', 'shard_dim("devices", -1)'),
            (r'llm/layers/mlp/linear', 'shard_dim("devices", -2)'),
            (r'llm/layers/attn/q_einsum/w', 'shard_dim("devices", -3)'),
            (r'llm/embedder/input_embedding', 'shard_dim("devices", -1)'),
            # pyformat: enable
            # pylint: enable=line-too-long
        ],
        mesh=_get_mesh(),
    )
  else:
    raise ValueError(f'Unknown sharding: {sharding_strategy}')
  params = jax.tree.map(utils.reshard, params_cpu, params_sharding)
  return params


def _pil2np(img):
  """Accepts `PIL.Image` or `np.ndarray` and returns `np.ndarray`."""
  if isinstance(img, PIL.Image.Image):
    img = np.array(img)
    img = img[..., :3]
    if img.ndim == 2:
      img = img[..., None]
    if img.shape[-1] == 1:
      img = np.repeat(img, 3, axis=-1)
  return img


@functools.cache
def _get_pp_fn(res, tokenizer_spec, text_len):
  return tf.function(
      lambda d: pp_builder.get_preprocess_fn(
          '|'.join([
              f'resize({res}, antialias=True)|value_range(-1, 1)',
              f"tok(key='prefix', bos='yes', model='{tokenizer_spec}')",
              f"tok(key='septok', text='\\n', model='{tokenizer_spec}')",
              f"tok(key='suffix', eos='yes', model='{tokenizer_spec}')",
              (
                  # pyformat: disable
                  'masked_concat(["prefix", "septok", "suffix"],'
                  ' mask_prefix=[1, 0, 0],'
                  ' mask_suffix=[0, 0, 1],'
                  ' mask_ar=[0, 0, 1],'
                  ' mask_input=[1, 1, 1])'
                  # pyformat: enable
              ),
              f'tolen({text_len}, pad_value=0, key="text")',
              f'tolen({text_len}, pad_value=0, key="mask_prefix")',
              f'tolen({text_len}, pad_value=0, key="mask_suffix")',
              f'tolen({text_len}, pad_value=1, key="mask_ar")',
              f'tolen({text_len}, pad_value=0, key="mask_input")',
              (
                  'keep("image", "text", "mask_prefix", "mask_suffix",'
                  ' "mask_ar", "mask_input")'
              ),
          ]),
          log_data=False,
      )({**d})
  )


def _prepare_batch(
    images,
    prefixes,
    *,
    res,
    tokenizer_spec,
    suffixes,
    text_len,
):
  """Returns non-sharded batch."""

  pp_fn = _get_pp_fn(res, tokenizer_spec, text_len)
  assert not isinstance(prefixes, str), f'expected batch: {prefixes}'
  assert (
      isinstance(images, (list, tuple)) or images.ndim == 4
  ), f'expected batch: {images.shape}'
  if suffixes is None:
    suffixes = [''] * len(prefixes)
  assert len(prefixes) == len(suffixes) == len(images), (
      f'invalid lengths: {len(prefixes)}, {len(suffixes)}, {len(images)}')
  examples = [{'_mask': True, **pp_fn({
      'image': np.asarray(_pil2np(image)),
      'prefix': np.array(prefix),
      'suffix': np.array(suffix),
  })} for image, prefix, suffix in zip(images, prefixes, suffixes)]
  batch = jax.tree.map(lambda *xs: np.stack(xs), *examples)
  return batch


def _shard_batch(batch, n=None):
  """Shards `batch` with fsdp strategy on all available devices."""
  if n is None:
    n = jax.local_device_count()
  def pad(x):
    return np.pad(x, [(0, -len(x) % n)] + [(0, 0)] * (x.ndim - 1))
  batch = {k: pad(v) for k, v in batch.items()}
  data_sharding = jax.sharding.NamedSharding(
      _get_mesh(), jax.sharding.PartitionSpec('devices')
  )
  batch_on_device = utils.reshard(batch, data_sharding)
  return batch_on_device


def _replicate_batch(batch):
  """Replicates `batch` on all available devices."""
  data_sharding = jax.sharding.NamedSharding(
      _get_mesh(), jax.sharding.PartitionSpec()
  )
  batch_on_device = utils.reshard(batch, data_sharding)
  return batch_on_device


@dataclasses.dataclass(frozen=True, kw_only=True, order=True)
class PaligemmaConfig:
  """Desribes a `big_vision` PaliGemma model."""

  ckpt: str
  params_dtype: str | None
  res: int
  llm_variant: str
  text_len: int
  tokenizer_spec: str
  vocab_size: int
  sharding_strategy: ShardingStrategy
  loading_params: dict[str, str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Score:
  """Score for a generation.

  Attributes:
    suffix: The text string that was scored.
    tokens: Tokens of `suffix` (shape `[seq_len]`).
    scores: Logprob of `tokens` (shape `[seq_len]`).
    logprobs: Optional log of probabilities (shape `[seq_len, vocab_size]`).
    pieces: String representation of individual `tokens` (shape `[seq_len]`).
  """
  suffix: str
  tokens: jnp.ndarray
  scores: jnp.ndarray
  logprobs: jnp.ndarray
  pieces: list[str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class PaliGemmaModel:
  """Wraps a `big_vision` PaliGemma model.

  Attributes:
    config: Config for the model.
    tokenizer: Tokenizer for the model.
    decode: Decode function for the model.
    beam_decode: Beam decode function for the model.
    logits: Logits function for the model.
  """

  config: PaligemmaConfig
  tokenizer: tokenizer.Tokenizer
  decode: Callable[..., Any]
  beam_decode: Callable[..., Any]
  logits: Callable[..., Any]

  def shard_batch(self, batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    """Shards `batch` according to `self.config.sharding_strategy`."""
    if self.config.sharding_strategy == ShardingStrategy.MEGATRON:
      return _replicate_batch(batch)
    else:
      return _shard_batch(batch)

  def shard_params(self, params_cpu):
    """Shards `params_cpu` according to `self.config.sharding_strategy`."""
    with _get_mesh(), nn.logical_axis_rules(_get_sharding_rules()):
      return _shard_params(params_cpu, self.config.sharding_strategy)

  def prepare_batch(
      self, images, prefixes, suffixes=None,
      *, text_len=None, check_overflow=True,
  ) -> dict[str, np.ndarray]:
    """Prepares a batch of data.

    Args:
      images: List of `PIL.Image.Image` or `np.ndarray` images.
      prefixes: The prefix (e.g. "caption en") will have a full attention mask
        with the image.
      suffixes: The suffix (e.g. "picture of a dog and") will be extended
        auto-regressively (thus with causal attention mask). This argument is
        usually omitted, but can be useful e.g. to continue a partial caption.
      text_len: Optional text length for `concat(prefix, suffix)`. Defaults to
        `self.config.text_len`.
      check_overflow: If true, will raise an error if the text length is too
        small to fit prefixes and suffixes.

    Returns:
      A dictionary of `tf.tensor()` as expected by the JAX model.
    """
    batch = _prepare_batch(
        images=images,
        prefixes=prefixes,
        suffixes=suffixes,
        res=self.config.res,
        tokenizer_spec=self.config.tokenizer_spec,
        text_len=text_len or self.config.text_len,
    )
    if check_overflow:
      padding = (1 - batch['mask_input']).sum(axis=-1)
      if not (padding > 0).all():
        raise ValueError(
            f'Missing padding: {padding} - consider setting larger `text_len`'
        )
    return batch

  def generate(
      self,
      params,
      batch,
      devices=None,
      max_decode_len=128,
      sampler='greedy',
      **kw,
  ):
    """Autoregressively generates tokens."""
    if devices is None:
      devices = _get_devices()
    if sampler == 'beam':
      decode = self.beam_decode
    else:
      decode = self.decode
      kw['sampler'] = sampler
    if (
        self.config.sharding_strategy == ShardingStrategy.MEGATRON
        and len(batch['text']) == 1
    ):
      kw['eos_look_behind'] = 1  # decreases latency with single ex inference
    with _get_mesh(), nn.logical_axis_rules(_get_sharding_rules()):
      return np.asarray(decode(
          {'params': params},
          batch=batch,
          devices=devices,
          eos_token=self.tokenizer.eos_token,
          max_decode_len=max_decode_len,
          **kw,
      ))

  def _score(self, params, batch, return_logprobs):
    """Returns scores and optionally logprobs."""
    logits, unused_out = self.logits(
        {'params': params},
        batch=batch,
    )
    # TODO: b/andstein - Check if it's worth to only compute logits for suffix.
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    scores = []
    for tokens, logprob in zip(batch['text'], logprobs):
      assert len(tokens) == len(logprob) + 1
      # logprobs predict NEXT token
      scores.append(logprob[np.arange(len(tokens)) - 1, tokens])
    if not return_logprobs:
      logprobs = jnp.zeros(logprobs.shape[:-1] + (0,), logprobs.dtype)
    return jnp.stack(scores), logprobs

  def score(self, params, batch, *, return_logprobs=False) -> list[Score]:
    """Returns scores for suffix.

    Args:
      params: Sharded parameters.
      batch: Sharded batch.
      return_logprobs: If true, will return logprobs in addition to scores (with
        a large vocabulary this might be relatively expensive).

    Returns:
      A list of `Score` objects (with the `logprobs` field set depending on the
      parameter `return_logprobs`). Note that the `Score` objects only account
      for the `suffix` part of the batch.
    """
    can_shard = len(batch['text']) % jax.device_count() == 0
    pargs = ('devices',) if can_shard else ()
    out_sharding = jax.sharding.NamedSharding(
        _get_mesh(), jax.sharding.PartitionSpec(*pargs)
    )
    with _get_mesh(), nn.logical_axis_rules(_get_sharding_rules()):
      all_scores, all_logprobs = jax.jit(
          self._score,
          static_argnames=['return_logprobs'],
          out_shardings=out_sharding,
      )(params, batch=batch, return_logprobs=return_logprobs)
    ret = []
    for mask_suffix, tokens, scores, logprobs in zip(*jax.device_get((
        batch['mask_suffix'],
        batch['text'],
        all_scores,
        all_logprobs,
    ))):
      assert mask_suffix.ndim == 1
      mask_suffix = mask_suffix == 1
      tokens = tokens[mask_suffix]
      scores = scores[mask_suffix]
      # logprobs predict NEXT token
      logprobs = logprobs[mask_suffix[1:]]
      if hasattr(self.tokenizer, 'to_piece'):
        pieces = [self.tokenizer.to_piece(t) for t in tokens]  # pytype: disable=attribute-error
      else:
        pieces = [self.tokenizer.to_str(t) for t in tokens]
      ret.append(
          Score(
              suffix=self.tokenizer.to_str(tokens),
              tokens=tokens,
              scores=scores,
              logprobs=logprobs,
              pieces=pieces,
          )
      )
    return ret


ParamsCpu = Any


def load_model(config: PaligemmaConfig) -> tuple[PaliGemmaModel, ParamsCpu]:
  """Loads model from config."""
  loader_path = config.loading_params.get('loader', None)
  if loader_path:
    loader_m = importlib.import_module(loader_path)
    return loader_m.load_model(config)
  load_f = _init_dummy if config.ckpt == DUMMY_MODEL else _load
  model, params_cpu, pred_fns, tok = load_f(
      path=config.ckpt,
      tokenizer_spec=config.tokenizer_spec,
      vocab_size=config.vocab_size,
      llm_variant=config.llm_variant,
  )
  if config.params_dtype:
    params_cpu = jax.tree.map(
        lambda x: x.astype(config.params_dtype), params_cpu
    )
  del model
  return PaliGemmaModel(
      config=config,
      tokenizer=tok,
      decode=pred_fns['decode'],
      beam_decode=pred_fns['beam_decode'],
      logits=pred_fns['logits'],
  ), params_cpu


_NAME_RE = re.compile(r'(\w+)(?:_(772m|2b|9b|27b))?_(224|448|896)(?:\..*)')


@functools.cache
def _infer_config(name, ckpt):
  """Returns inferred config parameters from name / checkpoint."""
  m = _NAME_RE.fullmatch(name)
  assert m, name  # tested in caller
  res = int(m.group(3))
  llm_variant = f'gemma2_{m.group(2)}' if m.group(2) else 'gemma_2b'

  res_name = name.split('_')[-1]
  if res_name in ('224', '448', '896'):
    res = int(res_name)

  ts_dir = None
  if tf.io.gfile.isdir(ckpt):
    ts_dir = ckpt
  elif tf.io.gfile.exists(f'{ckpt}-LAST'):
    num = tf.io.gfile.GFile(f'{ckpt}-LAST').read()
    ts_dir = f'{ckpt}-{num}'

  if ts_dir:
    posembed_zarr = f'{ts_dir}/params~img~pos_embedding/.zarray'
    if tf.io.gfile.exists(posembed_zarr):
      d = json.load(tf.io.gfile.GFile(posembed_zarr))
      n = d['shape'][1]
      res_in_patches = int(n ** 0.5)
      assert res_in_patches * res_in_patches == n, n
      res = _PATCH_SIZE * res_in_patches
    inputembed_zarr = f'{ts_dir}/params~llm~embedder~input_embedding/.zarray'
    if tf.io.gfile.exists(inputembed_zarr):
      d = json.load(tf.io.gfile.GFile(inputembed_zarr))
      unused_vocab, width = d['shape']
      llm_variant = {
          1152: 'gemma2_772m',
          2048: 'gemma_2b',
          2304: 'gemma2_2b',
          3584: 'gemma2_9b',
          4608: 'gemma2_27b',
      }[width]

  return res, llm_variant


def get_config(
    name_or_path: str,
    text_len: int = 256,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FSDP,
    params_dtype: str | None = None,
    res: int | None = None,
    llm_variant: str | None = None,
    tokenizer_spec: str = 'gemma(tokensets=("loc", "seg"))',
) -> PaligemmaConfig:
  """Returns config for model `name`."""
  if name_or_path == DUMMY_MODEL:
    name = ckpt = DUMMY_MODEL
    vocab_size = 32 + 1024 + 128
  else:
    if '/' in name_or_path:
      ckpt = name_or_path
      name = ''
      for name in ckpt.split('/')[::-1]:
        if _NAME_RE.fullmatch(name):
          break
      if not _NAME_RE.fullmatch(name):
        name = 'unknown_224'
    else:
      raise ValueError('Please provide full path containing "/".')
    vocab_size = 256_000 + 1024 + 128

  return PaligemmaConfig(
      ckpt=ckpt,
      params_dtype=params_dtype,
      res=res or _infer_config(name, ckpt)[0],
      llm_variant=llm_variant or _infer_config(name, ckpt)[1],
      text_len=text_len,
      tokenizer_spec=tokenizer_spec,
      vocab_size=vocab_size,
      sharding_strategy=sharding_strategy,
      loading_params={},
  )


def get_experimental_path(name: str) -> str | None:
  """Returns path to experimental model."""
  if re.fullmatch(r'[\w/]+', name):
    path = f'{EXPERIMENTAL_BASE}/{name}/checkpoint.bv'
    if tf.io.gfile.exists(f'{path}-LAST'):
      return path
