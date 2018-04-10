# remove warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#dependencies
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from main import env
from utils import *


# initializer = tf.contrib.layers.variance_scaling_initializer()
initializer = tf.contrib.layers.xavier_initializer()

def simple_nn():
	n_inputs = 4 # dynamic env.observation_space.size?ish?
	n_hidden = 4 
	n_out = 1

	X = tf.placeholder(tf.float32,shape=[None,n_inputs])
	hidden = fully_connected(X, n_hidden, activation_fn = tf.nn.elu, weights_initializer= initializer)
	logits = fully_connected(hidden, n_out, activation_fn = None, weights_initializer = initializer)
	out = tf.nn.sigmoid(logits)
	p_arr = tf.concat([out, 1 - out], axis = 1)
	action = tf.multinomial(tf.log(p_arr),num_samples = 1)
	y = 1 - tf.to_float(action)
	lr = 0.01
	loss_fn = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)
	opt = tf.train.AdamOptimizer(lr)

	grads_var = opt.compute_gradients(loss_fn)
	a_grads = [grads for grads, var in grads_var]
	grad_phs = []
	grads_var_feed = []
	for grads, var in grads_var:
		grad_ph = tf.placeholder(tf.float32,shape=grads.get_shape())
		grad_phs.append(grad_ph)
		grads_var_feed.append((grad_ph,var))
	training_op = opt.apply_gradients(grads_var_feed)

	init = tf.global_variables_initializer()
	save = tf.train.Saver()

	n_epoch = 2
	n_maxs = 1000
	update_every = 10
	save_every = 10
	drate = 0.95
# def train(n_epoch = 10, n_maxs = 1000, update_every = 10, save_every = 10, drate = 0.95):
	with tf.Session() as sess:
		init.run()
		for epoch in range(n_epoch):
			print(f'epoch {epoch}')
			all_rewards = []
			all_grads = []
			for game in range(update_every):
				current_rewards = []
				current_grads = []
				obs = env.reset()
				for step in range(n_maxs):
					action_val, gradients_val = sess.run([action, a_grads],feed_dict={X: obs.reshape(1, n_inputs)})
					obs, reward, done, info = env.step(action_val[0][0])
					current_rewards.append(reward)
					current_grads.append(gradients_val)
					if done:
						break
				all_rewards.append(current_rewards)
				all_grads.append(current_grads)
				print(all_grads.shape)
				break
			break

			all_rewards = normalize_rew(all_rewards,drate)
			feed_dict = {}
			for var_index, grad_placeholder in enumerate(grad_phs):
				mean_gradients = np.mean([reward * all_grads[game_index][step][var_index] for game_index,rewards in enumerate(all_rewards) for step,reward in enumerate(rewards)], axis=0)
				feed_dict[grad_ph] = mean_gradients

			sess.run(training_op, feed_dict=feed_dict)
			if iteration % save_every == 0:
				saver.save(sess, "./my_policy_net_pg_{epoch}.ckpt")






if __name__ == "__main__":
	simple_nn()
	# print(discount_rewards([10.,0,-50.],0.8))
	# print(normalize_rew([[10, 0, -50], [10, 20]], drate=0.8))
	# print("HIIII")

