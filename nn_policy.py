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



if __name__ == "__main__":
	simple_nn([2] * 4)


