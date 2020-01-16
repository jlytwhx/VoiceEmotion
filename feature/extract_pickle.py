# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

import pickle
from os import path

import numpy as np
import six
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim
from scipy.io import wavfile

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'settings/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'settings/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.wav_file:
        wav_file = FLAGS.wav_file
    else:
        num_secs = 5
        freq = 1000
        sr = 44100
        t = np.linspace(0, num_secs, int(num_secs * sr))
        x = np.sin(2 * np.pi * freq * t)
        samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
        wav_file = six.BytesIO()
        wavfile.write(wav_file, sr, samples)
        wav_file.seek(0)
    examples_batch = vggish_input.wavfile_to_examples(wav_file)
    # pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
    # print(FLAGS.checkpoint)
    # print(os.getcwd())
    # print(path.exists(FLAGS.checkpoint))
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        # embedding_batch为提取结果
        with open(path.splitext(wav_file)[0] + '.pk', "wb")as f:
            pickle.dump(embedding_batch, f)


if __name__ == '__main__':
    tf.app.run()
