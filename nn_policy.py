import numpy as np
import pdb
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# initializer = tf.contrib.layers.variance_scaling_initializer()
initializer = tf.contrib.layers.xavier_initializer()

def simple_nn(observation):
	n_inputs = len(observation) 
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
	traning_op = opt.apply_gradients(grads_var_feed)

	init = tf.global_variable_initializer()
	save = tf.train.saver()



	# print(out)

def discount_rewards(rewards,drate = 0.8):
	l = len(rewards)
	rewards = np.array(rewards)
	drates = np.array([drate**i for i in range(l)])
	dreward = np.array([np.sum(rewards[i:] * drates[:l-i]) for i in range(l)])
	return dreward

def normalize_rew(all_rewards, drate = 0.8):
	# pdb.set_trace()
	all_discounted_rewards = [discount_rewards(rewards) for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	reward_std = flat_rewards.std()
	return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]



if __name__ == "__main__":
	# simple_nn([2] * 4)
	# print(discount_rewards([10.,0,-50.],0.8))
	# print(normalize_rew([[10, 0, -50], [10, 20]], drate=0.8))
	print("HIIII")

