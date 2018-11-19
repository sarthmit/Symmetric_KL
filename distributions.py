import tensorflow as tf

def forward_gaussian(X, log_var, mean):
	return tf.reduce_mean(0.5*tf.reduce_sum(log_var+tf.exp(-log_var)*((mean - X)**2.),axis=1))

def forward_gamma(param_1_log, param_2_log):
	return tf.reduce_mean(tf.reduce_sum(-tf.math.lgamma(tf.exp(param_1_log)) + tf.exp(param_1_log)*param_2_log + (tf.exp(param_1_log) - 1)*X - tf.exp(param_2_log)*X), axis=1)


def forward_dirichlet():


def forward_beta():


def forward_kumaraswamy(X, param_1_log, param_2_log):
	# Requires input to lie in [0,1]

	return tf.reduce_mean(tf.reduce_sum(param_1_log + param_2_log + (tf.exp(param_1_log)-1)*tf.log(X) + (tf.exp(param_2_log)-1)*(1-tf.pow(x,tf.exp(param_1_log))), axis=1))