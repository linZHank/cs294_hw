#!/usr/bin/env python

"""
Code to train behavioral cloning agents based on expert data.
Example usage:

Author of this script and included expert policies: linZHank (linzhank@gmail.com)
"""

import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util

tf.enable_eager_execution()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("expert_data_file", type=str)
  args = parser.parse_args()
  
  with open(args.expert_data_file, "rb") as f:
    expert_data = pickle.load(f)
  observations = expert_data["observations"].astype(np.float32)
  actions = expert_data["actions"]
  actions = actions.reshape(actions.shape[0], actions.shape[-1]).astype(np.float32)

  # train behavioral cloning policy based on expert data
  # training parameter
  num_epochs = 1000
  batch_size = 10000
  learning_rate = 0.01
  # model, dataset, optimizer, loss, etc..
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(observations.shape[1],)), 
    tf.keras.layers.Dense(actions.shape[-1])
  ])
  dataset = tf_util.create_dataset(
    input_features=observations,
    output_labels=actions,
    batch_size=batch_size,
    num_epochs=num_epochs
  )
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  global_step = tf.train.get_or_create_global_step()
  loss_value, grads = tf_util.grad(
    model,
    observations,
    actions
  )

  # train nn
  for i, (x, y) in enumerate(dataset):
    # optimize model
    loss_value, grads = tf_util.grad(model, x, y)
    optimizer.apply_gradients(
      zip(grads, model.variables),
      global_step
    )
    # log training
    if not i % 100:
      print("Iteration: {}, Loss: {:.3f}".format(i, loss_value))
