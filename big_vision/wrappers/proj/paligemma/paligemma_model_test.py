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

from absl.testing import absltest
from big_vision.wrappers.proj.paligemma import paligemma_model
import jax
import numpy as np


class ModelTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ModelTest, cls).setUpClass()
    assert jax.device_count() == 1, 'Tests assume single device.'
    jax.config.update('jax_transfer_guard', 'disallow')
    import pathlib
    path = pathlib.Path(__file__).parent.resolve() / 'test_tokenizer.model'
    tokenizer_spec = f'gemma(model="{path}", tokensets=("loc", "seg"))'
    cls.config = paligemma_model.get_config(
        paligemma_model.DUMMY_MODEL, tokenizer_spec=tokenizer_spec
    )
    cls.model, cls.params_cpu = paligemma_model.load_model(cls.config)
    cls.params = cls.model.shard_params(cls.params_cpu)

  def test_dummy_model_params(self):
    self.model, self.params_cpu = paligemma_model.load_model(self.config)
    self.assertEqual(
        np.sum(np.prod(x.shape) for x in jax.tree.leaves(self.params)), 147_648
    )

  def test_prepare_batch(self):
    batch = self.model.prepare_batch(
        images=[np.zeros([1, 2, 3]), np.zeros([2, 1, 3])],
        prefixes=['test1', 'test2'],
    )
    self.assertIn('image', batch)
    self.assertEqual(batch['image'].shape, (2, 224, 224, 3))
    self.assertIn('text', batch)
    self.assertEqual(batch['text'].shape, (2, self.config.text_len))
    with self.assertRaisesRegex(ValueError, 'Missing padding'):
      _ = self.model.prepare_batch(
          images=[np.zeros([1, 1, 3])],
          prefixes=['blah ' * 1024],
      )

  def test_generate(self):
    batch = self.model.prepare_batch(
        images=[np.zeros([1, 1, 3])],
        prefixes=[''],
    )
    batch = self.model.shard_batch((batch))
    tokens = self.model.generate(self.params, batch, max_decode_len=12)
    self.assertEqual(tokens.shape, (1, 12))

  def test_score(self):
    batch = self.model.prepare_batch(
        images=[np.zeros([1, 1, 3])],
        prefixes=['test prefix'],
        suffixes=['test suffix']
    )
    batch = self.model.shard_batch((batch))
    scores = self.model.score(self.params, batch, return_logprobs=True)
    self.assertLen(scores, 1)
    score = scores[0]
    self.assertEqual(scores[0].suffix, 'test suffix')
    n = len(score.tokens)
    self.assertLen(score.scores, n)
    self.assertLen(score.logprobs, n)
    self.assertLen(score.pieces, n)
    for i in range(n):
      self.assertLess(np.exp(score.logprobs[i]).sum() - 1, 1e-3)


if __name__ == '__main__':
  absltest.main()
