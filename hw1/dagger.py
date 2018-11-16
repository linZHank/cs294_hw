#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3 run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
from __future__ import absolute_import, division, print_function

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

# def main():
    
#     print('loading and building expert policy')
#     policy_fn = load_policy.load_policy(args.expert_policy_file)
#     print('loaded and built')

#     with tf.Session():
#         tf_util.initialize()

#         import gym
#         env = gym.make(args.envname)
#         max_steps = args.max_timesteps or env.spec.timestep_limit

#         returns = []
#         observations = []
#         actions = []
#         for i in range(args.num_rollouts):
#             print('iter', i)
#             obs = env.reset()
#             done = False
#             totalr = 0.
#             steps = 0
#             while not done:
#                 action = policy_fn(obs[None,:])
#                 observations.append(obs)
#                 actions.append(action)
#                 obs, r, done, _ = env.step(action)
#                 totalr += r
#                 steps += 1
#                 if args.render:
#                     env.render()
#                 if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
#                 if steps >= max_steps:
#                     break
#             returns.append(totalr)

#         print('returns', returns)
#         print('mean return', np.mean(returns))
#         print('std of return', np.std(returns))

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_data_file', type=str)
  # parser.add_argument('envname', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument("--max_timesteps", type=int)
  parser.add_argument('--num_rollouts', type=int, default=20,
                      help='Number of expert roll outs')
  args = parser.parse_args()

  # Load expert_data: (obs, actions)
  with open(args.expert_data_file, "rb") as f:
    expert_data = pickle.load(f)
  expert_obss = expert_data["observations"].astype(np.float32)
  expert_acts = expert_data["actions"].reshape(expert_obss.shape[0], -1).astype(np.float32)

  # Behavioral clone
  # normalize
  epsilon = 1e-12
  expert_obss_range = expert_obss.ptp(axis=0)
  expert_obss_range = np.array([max(epsilon, vr) for vr in expert_obss_range]).astype(np.float32)
  expert_obss_min = expert_obss.min(axis=0)
  normalized_expert_obss = (expert_obss-expert_obss_min)/expert_obss_range
  # dataset parameters
  batch_size = 5000
  num_epochs = 100
  learning_rate = 1e-2
  # make dataset use expert data
  expert_dataset = tf_util.create_dataset(
    input_features=normalized_expert_obss,
    output_labels=expert_acts,
    batch_size=batch_size,
    num_epochs=num_epochs
  )
  iterator = expert_dataset.make_one_shot_iterator()
  next_obss, next_acts = iterator.get_next()
  # define model
  bc_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(expert_obss.shape[1],)),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(expert_acts.shape[-1])
  ])
  # define training
  loss = tf_util.loss(bc_model, next_obss, next_acts)
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss)
  # train cloned policy
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    i = 0
    while True:
      try:
        # (next_obss, next_acts) = sess.run(iterator.get_next())
        # loss_value = sess.run(loss)
        _, loss_value = sess.run([train_op, loss])
        print(i, "\n", loss_value)
      except tf.errors.OutOfRangeError:
        break
      i += 1
          
  # with tf.Session() as sess:
  #   for e in range(20):
  #     # sess.run(iterator.initializer)
  #     i = 0
  #     while True:
  #       try:
  #         na = sess.run(next_act)
  #         print(e, ",", i, "\n")
  #       except tf.errors.OutOfRangeError:
  #         break
  #       i += 1
  # # Load behavioral cloning policy
  

  # Run BC policy and collect data: (obs)

  # Label newly collected data with expert policy_fn

  # Data aggregate

  

