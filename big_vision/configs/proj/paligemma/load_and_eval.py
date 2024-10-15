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

r"""Loads a model and runs evaluations.

If you want to profile an evaluator, add a line like this into its loop::

  sess = u.startstop_prof_at_steps(sess, step, 1, 11, name='...')
"""

# pylint: disable=line-too-long

import re

import big_vision.configs.common as bvcc
from big_vision.configs.proj.paligemma.transfers import ai2d as transfer_lib


RELEASE_BASE = '.'


def get_config(arg=None):
  """Returns an eval-only config."""

  arg = bvcc.parse_arg(
      arg,
      model_init='pt_224',
      sharding='fsdp',
      batch_size=256,
      res=224,
      evaluator_re='.*',
  )
  c = transfer_lib.get_config(f'res={arg.res}')
  c.update(arg)

  if '/' not in c.model_init:  # copybara: strip
    c.model_init = (  # copybara: strip
        f'{RELEASE_BASE}/{c.model_init}/checkpoint.bv'
    )

  for suffix in ('steps', 'examples', 'epochs'):
    if f'total_{suffix}' in c:
      del c[f'total_{suffix}']
  c.total_steps = 0

  for k in list(c.evals):
    if re.fullmatch(c.evaluator_re, k):
      c.evals[k].batch_size = c.batch_size
    else:
      del c.evals[k]

  text_len = 1  # only used for model.init()
  c.input = dict(
      data=dict(
          name='bv:dummy',
          spec=dict(
              image=dict(shape=(arg.res, arg.res, 3), dtype='float32'),
              text=dict(shape=(text_len,), dtype='int32'),
              mask_ar=dict(shape=(text_len,), dtype='int32'),
          ),
      )
  )
  c.input.batch_size = c.eval_batch_size = c.batch_size
  c.optax_name = 'identity'
  c.optax = {}

  c.sharding_rules = [('act_batch', ('devices',))]
  c.mesh = [('devices', -1)]
  if c.sharding == 'fsdp':
    c.sharding_strategy = [('.*', 'fsdp(axis="devices")')]
  elif c.sharding == 'megatron':
    c.sharding_strategy = [
        # pyformat: disable
        (r'.*/img/Transformer/encoderblock/MlpBlock_\d+/Dense_0/kernel', 'shard_dim("devices", -1)'),
        (r'.*/img/Transformer/encoderblock/MlpBlock_\d+/Dense_1/kernel', 'shard_dim("devices", -2)'),
        (r'.*/img/Transformer/encoderblock/MultiHeadDotProductAttention_\d+/(query|key|value)/kernel', 'shard_dim("devices", -2)'),
        (r'.*/img/Transformer/encoderblock/MultiHeadDotProductAttention_\d+/out/kernel', 'shard_dim("devices", -3)'),
        (r'.*/llm/layers/mlp/gating_einsum', 'shard_dim("devices", -1)'),
        (r'.*/llm/layers/mlp/linear', 'shard_dim("devices", -2)'),
        (r'.*/llm/layers/attn/q_einsum/w', 'shard_dim("devices", -3)'),
        (r'.*/llm/embedder/input_embedding', 'shard_dim("devices", -1)'),
        # pyformat: enable
    ]
  else:
    raise ValueError(f'Unknown sharding: {c.sharding}')

  c.name = ''
  return c