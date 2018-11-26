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
def normalize(observation, observations_buffer):
  """
  Normalize current observation with range and minimum of observations buffer
  """
  epsilon = 1e-12
  obss_range = np.array(observations_buffer).ptp(axis=0)
  obss_range = np.array([max(epsilon, vr) for vr in obss_range]).astype(np.float32)
  obss_min = np.array(observations_buffer).min(axis=0)
  normalized_observation = (observation-obss_min)/obss_range

  return normalized_observation
  
  
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_policy_file', type=str)
  parser.add_argument('envname', type=str)
  parser.add_argument('--expert_data', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument("--max_timesteps", type=int)
  parser.add_argument('--num_rollouts', type=int, default=20,
                      help='Number of expert roll outs')
  args = parser.parse_args()

  # Load expert policy 
  print("Loading and building expert policy")
  expert_policy_fn = load_policy.load_policy(args.expert_policy_file)
  print("Expert policy loaded and built")
  
  # Load expert data
  agg_observations = []
  agg_actions = []
  print("Loading expert policy generated data: obs and actions pair")
  with open(args.expert_data, "rb") as f:
    expert_data = pickle.load(f)
  assert expert_data["observations"].shape[0] == expert_data["actions"].shape[0]
  for i in range(expert_data["observations"].shape[0]):
    agg_observations.append(expert_data["observations"][i])
    agg_actions.append(expert_data["actions"][i,-1])

  # Restore model
  learning_rate = 1e-3
  bc_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(expert_data["observations"].shape[1],)),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(expert_data["actions"].shape[-1])
  ])
  # define training
  loss = tf_util.loss(bc_model, next_obss, next_acts)
  optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss)
  model_path = "./bc_model/model.ckpt"
  saver = tf.train.Saver()
  normobs_placeholder = tf.placeholder(tf.float32, shape=(None, expert_data["observations"].shape[-1]))
  learned_action = bc_model(normobs_placeholder)
  
  # Run BC policy and collect data: (obs)    
  env = gym.make(args.envname)
  max_steps = args.max_timesteps or env.spec.timestep_limit
  episodic_rewards = []
  for rollout in range(args.num_rollouts):
    for episode in range(2): # train model every 2 episodes
      obs = env.reset()
      normed_obs = normalize(obs, agg_observations)
      done = False
      total_reward = 0
      step = 0
      with tf.Session() as sess:
        saver.restore(sess, model_path)
        while not done:
          action = sess.run(learned_action, feed_dict={normobs_placeholder: normed_obs[None,:]})
          expert_action = expert_policy_fn(obs[None,:])
          agg_observations.append(obs)
          agg_actions.append(expert_action)
          obs, reward, done, _ = env.step(action)
          normed_obs = normalize(obs, agg_observations)
          total_reward += reward
          step += 1
          if args.render:
            env.render()
          if not step % 10:
            print("Episode: {}, Step: {} of {}, total reward: {}".format(episode, step, max_steps, total_reward))
          if step >= max_steps:
            break
      episodic_rewards.append(total_reward)
      # craete dataset based on 
      agg_dataset = tf.util.create_dataset(
        input_features=normalized_expert_obss,
        output_labels=expert_acts,
        batch_size=batch_size,
        num_epochs=num_epochs
      )
      
      # Train model with aggregated data
    # while True:
    #   try:
    #     # (next_obss, next_acts) = sess.run(iterator.get_next())
    #     # loss_value = sess.run(loss)
    #     _, loss_value = sess.run([train_op, loss])
    #     if not i%10:
    #       print("Iteration: {}, Loss: {:.3f}".format(i, loss_value))
    #   except tf.errors.OutOfRangeError:
    #     break
    #   i += 1
      
  #   print(
  #     "\nEpisodic returns: {}".format(episodic_returns),
  #     "\nAverage of the returns: {}".format(np.mean(episodic_returns)),
  #     "\nStandard deviation of the returns: {}".format(np.std(episodic_returns))
  #   )
  # Label newly collected data with expert policy_fn

  # Data aggregate  

