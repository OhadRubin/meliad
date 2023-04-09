# Copyright 2022 Google.
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

r"""Program to run a transformer model over a single article.

"""

# This program is currently a template, which can be expanded to do more
# sophisticated analysis.
import os
if "DEBUG" in os.environ:
  import debugpy
  debugpy.listen(5678)
  print("Waiting for debugger attach")
  debugpy.wait_for_client()
import tensorflow as tf
from gin import config
config.register_file_reader(tf.io.gfile.GFile, tf.io.gfile.exists)
from typing import Sequence
import numpy as np
from absl import app
from absl import flags
from clu import platform
import jax
from transformer import inference_utils
from transformer import tasks  # pylint: disable=unused-import
import tensorflow.compat.v2 as tf


flags.DEFINE_string("workdir", "", "Directory to save model checkpoints.")
flags.DEFINE_string("load_dir", "", "Directory to load pre-trained model.")
flags.DEFINE_integer("num_steps", None, "Number of steps.",required=True)

flags.DEFINE_list(
    "gin_search_paths",
    ["transformer/configs"],
    "List of paths where the Gin config files are located.")
flags.DEFINE_multi_string(
    "gin_file", ["base_htrans.gin"], "List of Gin config files.")
flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings.")
flags.DEFINE_enum(
    "split",None, ["validation", "test"],help="Which split to use.",required=True)
    
import json
FLAGS = flags.FLAGS

import tqdm
def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f"process_index: {jax.process_index()}, "
                                       f"process_count: {jax.process_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")

  inference_utils.parse_gin_configuration(FLAGS.gin_file, FLAGS.gin_param,
                                          gin_paths=FLAGS.gin_search_paths)
  losses_list = []
  task, task_state = None,None
  shape_set = set()
  loss = 0
  n_tokens = 0
  for article_data in tqdm.tqdm(inference_utils.read_article(FLAGS.split,verbose=False)):
    (_, vocab) = article_data
    if task is None:
      (task, task_state, _) = inference_utils.create_model_and_task(
          vocab, load_dir=FLAGS.load_dir)
    outs,task_state = inference_utils.run_model(task, task_state, article_data,
                                    verbose=False)
    losses = inference_utils.get_token_losses(outs)[0]
    losses = losses[losses!=0]
    shape=int(losses.shape[0])
    if shape not in shape_set:
      shape_set.add(shape)
      losses_list.append((losses.sum(),shape,))
      loss = loss + losses.sum()
      n_tokens = n_tokens + shape
    
      print("Perplexity : ", np.exp(loss/n_tokens))
      print("log2 Perplexity : ", np.log2(np.exp(loss/n_tokens)))
      print("Total loss: ", loss)
      print("Total nonzero tokens: ", n_tokens)
      if len(shape_set)==FLAGS.num_steps:
        break
  loss,n_tokens = zip(*losses_list)
  loss = np.sum(loss)
  n_tokens = np.sum(n_tokens)

  print("Perplexity : ", np.exp(loss/n_tokens))
  print("Total loss: ", loss)
  print("Total nonzero tokens: ", n_tokens)

if __name__ == "__main__":
  app.run(main)