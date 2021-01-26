# The following source code is obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L65-L166
# ==============================================================================

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Various learning rate decay functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule

class ExponentialDecay(learning_rate_schedule.LearningRateSchedule):
  """A LearningRateSchedule that uses an exponential decay schedule.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies an exponential decay function
  to an optimizer step, given a provided initial learning rate.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ^ (step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `step / decay_steps` is
  an integer division and the decayed learning rate follows a
  staircase function.

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate.
  Example: When fitting a Keras model, decay every 100000 steps with a base
  of 0.96:

  ```python
  initial_learning_rate = 0.1
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=100000,
      decay_rate=0.96,
      staircase=True)

  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(data, labels, epochs=5)
  ```

  The learning rate schedule is also serializable and deserializable using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      warmup_steps,
      decay_steps,
      decay_rate,
      staircase=False,
      name=None):
    """Applies exponential decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      decay_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The decay rate.
      staircase: Boolean.  If `True` decay the learning rate at discrete
        intervals
      name: String.  Optional name of the operation.  Defaults to
        'ExponentialDecay'.
    """
    super(ExponentialDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.staircase = staircase
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "ExponentialDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      warmup_steps = math_ops.cast(self.warmup_steps, dtype)
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      decay_rate = math_ops.cast(self.decay_rate, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      p = global_step_recomp / warmup_steps
      p = math_ops.maximum(p, K.epsilon())
      warmup_learning_rate = math_ops.multiply(
          initial_learning_rate, math_ops.pow(p, 1.0))

      global_step_recomp = math_ops.cast(step - self.warmup_steps, dtype)
      p = global_step_recomp / decay_steps
      if self.staircase:
        p = math_ops.floor(p)
      learning_rate = math_ops.multiply(
          initial_learning_rate, math_ops.pow(decay_rate, p))
      return control_flow_ops.cond(
          step < self.warmup_steps,
          lambda: warmup_learning_rate,
          lambda: learning_rate,
          name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "warmup_steps": self.warmup_steps,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "staircase": self.staircase,
        "name": self.name
    }
