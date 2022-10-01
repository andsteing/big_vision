# Copyright 2022 Big Vision Authors.
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

"""Text-centric preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of (TF) tensors. The functors output a modified
dictionary.

A commonly used key for the tokenized output is "labels".
"""
from absl import logging
from big_vision.datasets.imagenet import class_names as imagenet_class_names
from big_vision.pp.registry import Registry
import big_vision.pp.utils as utils
import tensorflow as tf
from tensorflow.io import gfile
import tensorflow_text as tftext


KNOWN_TOKENIZERS = {  # ref: go/webli-tok
    "c4_en": "gs://big_vision_eu/lit_private/c4_en/sentencepiece.model",
}


def create_tokenizer(model="c4_en", add_eos=True):
  logging.info("Creating tokenizer: %s", model)
  with gfile.GFile(KNOWN_TOKENIZERS.get(model, model), "rb") as f:
    model = f.read()
  return tftext.SentencepieceTokenizer(model=model, add_eos=add_eos)


def tokenize(input_text, tokenizer, max_len, *, pad_value, force_eos,
             multi_text=False):
  """Tokenizes string, and adds `pad_value` if longer than `max_len`."""

  def pad(tokens):
    # Truncate/pad to max_len.
    if force_eos:
      tokens = tf.cond(
          tf.shape(tokens)[0] >= max_len,
          lambda: tf.concat(  # pylint: disable=g-long-lambda
              # For too long, cut them off, but do keep the final EOS token.
              [tokens[:max_len - 1], tokens[-1:]], axis=0),
          lambda: tf.pad(  # pylint: disable=g-long-lambda
              tokens, [(0, max_len - tf.shape(tokens)[0])],
              constant_values=pad_value),
      )
    else:
      tokens = tokens[:max_len]
      tokens = tf.pad(
          tokens, [(0, max_len - tf.shape(tokens)[0])],
          constant_values=pad_value)
    tokens.set_shape([max_len])
    return tokens

  tokens = tokenizer.tokenize(input_text)
  if multi_text:
    tokens = tokens.to_tensor(pad_value)  # tf.RaggedTensor to tf.Tensor
    tokens = tf.reshape(tokens, [-1, tf.shape(tokens)[-1]])
    tokens = tf.map_fn(pad, tokens)  # `map_fn` only maps on axis 0
    return tf.reshape(tokens, tf.concat([tf.shape(input_text), [-1]], axis=0))
  else:
    return pad(tokens)


@Registry.register("preprocess_ops.tokenize")
@utils.InKeyOutKey(indefault=None, outdefault="labels")
def get_pp_tokenize(
    max_len,
    eos,
    model="c4_en",
    lower=True,
    sample_if_multi=True,
    pad_value="<pad>",
):
  """Tokenizes a text.

  Let's assume max_len=3 and id("</s>")=1, id("a")=2, then we have

  1. `eos="none", pad_value=0`:
     - "a" -> [2, 0, 0]
     - "aa" -> [2, 2, 0]
     - "aaa" -> [2, 2, 2]

  2. `eos="yes", pad_value=0`:
     - "a" -> [2, 1, 0]
     - "aa" -> [2, 2, 1]
     - "aaa" -> [2, 2, 2]

     This is usually used with generative models that need to learn when to
     properly predict a "</s>" (when the sentence is finished) and when to
     abstain (when the sentence is truncated).

  3. `eos="sticky", pad_value=0`:
     - "a" -> [2, 1, 0]
     - "aa" -> [2, 2, 1]
     - "aaa" -> [2, 2, 1]

  4. `eos="sticky", pad_value=1`:
     - "a" -> [2, 1, 1]
     - "aa" -> [2, 2, 1]
     - "aaa" -> [2, 2, 1]

     This is traditionally used with Argus discriminative models that simply
     refer to the last token for embeddings, similarly to how "cls" tokens are
     used in BERT-style models.

  Args:
    max_len: maximum length of the tokenized text.
    eos: Whether to add an "</s>" (end of sentence) token and whether to keep it
      when the sequence is longer than `max_len - 1`. See examples above for
      details. Valid values: "none", "yes", "sticky".
    model: a path to the pretrained sentencepiece model.
    lower: lowercase the text before tokenizing.
    sample_if_multi: If there's more than one, randomly pick one if this is
      True; otherwise pick all texts and keep the input's batch shape in result.
    pad_value: which token to pad the sequence with. If a string (for example
      `"<pad>"`), tokenize it and use its first token. Note that there is no
      guarantee to have any padding at the end of the sentence, if the sentence
      is longer than `max_len`.

  Returns:
    an op that outputs tokenized text.
  """

  if eos not in ("yes", "none", "sticky"):
    raise ValueError(f"Invalid value for eos: '{eos}'.")

  tokenizer = create_tokenizer(model, add_eos=eos != "none")

  if isinstance(pad_value, str):
    pad_value = tokenizer.string_to_id(pad_value)

  def _pp_tokenize(txt):
    if sample_if_multi:
      txt = tf.reshape(txt, (-1,))
      # Append an empty string so we gracefully handle empty cases.
      txt = tf.concat([txt, [""]], axis=0)
      num_texts = tf.maximum(tf.shape(txt)[0] - 1, 1)  # Don't sample "".
      txt = txt[tf.random.uniform([], 0, num_texts, dtype=tf.int32)]

    if lower:
      txt = tf.strings.lower(txt) if sample_if_multi else tf.map_fn(
          tf.strings.lower, txt)

    return tokenize(
        txt,
        tokenizer,
        max_len,
        pad_value=pad_value,
        force_eos=eos == "sticky",
        multi_text=not sample_if_multi)

  return _pp_tokenize


@Registry.register("preprocess_ops.clip_i1k_label_names")
@utils.InKeyOutKey(indefault="label", outdefault="labels")
def get_pp_clip_i1k_label_names():
  """Convert i1k label numbers to strings, using CLIP's class names."""

  def _pp_imagenet_labels(label):
    return tf.gather(imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES, label)

  return _pp_imagenet_labels
