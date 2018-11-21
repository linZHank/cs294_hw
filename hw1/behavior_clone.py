#!/usr/bin/env python

"""
Code to train behavioral cloning agents based on expert data.
Example usage:
python behavior_clone.py expert_data/Reacher-v2.pkl Reacher-v2 --render \
            --num_rollouts 10
Make sure: python run_expert.py first to collect expert's policy

Author of this script and included expert policies: linZHank (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function
import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_data_file', type=str)
  parser.add_argument('envname', type=str)
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
  batch_size = 40000
  num_epochs = 400
  learning_rate = 1e-3
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
  obs_placeholder = tf.placeholder(tf.float32, shape=(None, expert_obss.shape[-1]))
  cloned_action = bc_model(obs_placeholder)
  # define training
  loss = tf_util.loss(bc_model, next_obss, next_acts)
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss)
  # train cloned policy
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(init)
    i = 0
    while True:
      try:
        # (next_obss, next_acts) = sess.run(iterator.get_next())
        # loss_value = sess.run(loss)
        _, loss_value = sess.run([train_op, loss])
        if not i%10:
          print("Iteration: {}, Loss: {:.3f}".format(i, loss_value))
      except tf.errors.OutOfRangeError:
        break
      i += 1
    # save trained model
    save_path = saver.save(sess, "./bc_model/model.ckpt")
    print("Model saved in path : {}".format(save_path))
  # Apply trained policy
    env = gym.make(args.envname)
    # env parameters
    max_steps = args.max_timesteps or env.spec.timestep_limit
    episodic_returns = []
    observations = []
    actions = []
    for episode in range(args.num_rollouts):
      obs = env.reset()
      obs_norm = (obs-expert_obss_min) / expert_obss_range
      done = False
      total_reward = 0.
      step = 0
      while not done:
        # action = sess1.run(bc_model(obs_norm[None,:]))
        action = sess.run(cloned_action, feed_dict={obs_placeholder: obs_norm[None,:]})
        obs, reward, done, _ = env.step(action)
        obs_norm = (obs-expert_obss_min) / expert_obss_range
        total_reward += reward
        step += 1
        if args.render:
          env.render()
        # if not step % 100:
        print("Episode: {}, Step:{} of {}, reward: {}".format(episode, step, max_steps, reward))
        if step >= max_steps:
          break
      # append episodic return
      episodic_returns.append(total_reward)
    # log validation
    print(
      "\nEpisodic returns: {}".format(episodic_returns),
      "\nAverage of the returns: {}".format(np.mean(episodic_returns)),
      "\nStandard deviation of the returns: {}".format(np.std(episodic_returns))
    )
