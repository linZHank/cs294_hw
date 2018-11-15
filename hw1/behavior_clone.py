#!/usr/bin/env python

"""
Code to train behavioral cloning agents based on expert data.
Example usage:
python behavior_clone.py expert_data/Reacher-v2.pkl Reacher-v2 --render \
            --num_rollouts 10
Make sure: run run_expert.py first to collect expert's policy

Author of this script and included expert policies: linZHank (linzhank@gmail.com)
"""

import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym

tf.enable_eager_execution()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("expert_data_file", type=str)
  parser.add_argument('envname', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument("--max_timesteps", type=int)
  parser.add_argument('--num_rollouts', type=int, default=20,
                      help='Number of expert roll outs')
  args = parser.parse_args()
  
  with open(args.expert_data_file, "rb") as f:
    expert_data = pickle.load(f)
  observations = expert_data["observations"].astype(np.float32)
  epsilon = 1e-12
  values_range = observations.ptp(axis=0)
  values_range = np.array([max(epsilon, vr) for vr in values_range]).astype(np.float32)
  values_min = observations.min(axis=0)
  normalized_observations = (observations-values_min)/values_range
  actions = expert_data["actions"]
  actions = actions.reshape(actions.shape[0], actions.shape[-1]).astype(np.float32)

  # train behavioral cloning policy based on expert data
  # training parameter
  num_epochs = 200
  batch_size = 20000
  learning_rate = 1e-4
  # model, dataset, optimizer, loss, etc..
  bc_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(observations.shape[1],)),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(actions.shape[-1])
  ])
  dataset = tf_util.create_dataset(
    input_features=normalized_observations,
    output_labels=actions,
    batch_size=batch_size,
    num_epochs=num_epochs
  )
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  global_step = tf.train.get_or_create_global_step()
  loss_value, grads = tf_util.grad(
    bc_model,
    observations,
    actions
  )

  # train nn
  for i, (x, y) in enumerate(dataset):
    # optimize model
    loss_value, grads = tf_util.grad(bc_model, x, y)
    optimizer.apply_gradients(
      zip(grads, bc_model.variables),
      global_step
    )
    # log training
    if not i % 100:
      print("Iteration: {}, Loss: {:.3f}".format(i, loss_value))
  print("Iteration: {}, Loss: {:.3f}".format(i, loss_value)) # print last iter loss

  # apply trained policy
  env = gym.make(args.envname)
  max_steps = args.max_timesteps or env.spec.timestep_limit
  episodic_returns = []
  observations = []
  actions = []
  for episode in range(args.num_rollouts):
    # print("Episode: {}".format(episode))
    obs = env.reset()
    obs_norm = (obs-values_min) / values_range
    done = False
    total_reward = 0.
    step = 0
    while not done:
      action = bc_model(obs.reshape(1,obs.shape[0]))
      obs, reward, done, _ = env.step(action)
      obs_norm = (obs-values_min) / values_range
      total_reward += reward
      step += 1
      if args.render:
        env.render()
      # if not step % 100:
      print("Episode: {}, Step:{} of {}, reward: {}".format(episode, step, max_steps, reward))
      if step >= max_steps:
        break
      
    episodic_returns.append(total_reward)

  print(
    "\nEpisodic returns: {}".format(episodic_returns),
    "\nAverage of the returns: {}".format(np.mean(episodic_returns)),
    "\nStandard deviation of the returns: {}".format(np.std(episodic_returns))
  )
