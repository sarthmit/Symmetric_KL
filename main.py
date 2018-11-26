import os
import config
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn import mixture
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from sklearn.manifold import TSNE

slim = tf.contrib.slim
ds = tf.contrib.distributions
graph_replace = tf.contrib.graph_editor.graph_replace
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

tf.logging.set_verbosity(tf.logging.ERROR)

ds = tf.contrib.distributions
lrelu = tf.nn.leaky_relu
xav_init = tf.contrib.layers.xavier_initializer()

mnist = tf.contrib.learn.datasets.mnist.load_mnist(train_dir="mnist_data")

def encoder(X):    
    with tf.variable_scope("Encoder"):
        h_encoders = [
            tf.layers.dense(
                X,
                config.encoder_hidden_size[0],
                activation=tf.nn.relu,
                kernel_initializer=xav_init,
                name="encoder_hidden_layer_0",
                reuse=tf.AUTO_REUSE
            )
        ]
        
        for index, size in enumerate(config.encoder_hidden_size[1:]):
            h_encoders.append(
                tf.layers.dense(
                    h_encoders[index],
                    size,
                    activation=tf.nn.relu,
                    kernel_initializer=xav_init,
                    name="encoder_hidden_layer_" + str(index + 1),
                    reuse=tf.AUTO_REUSE
                )
            )
        
        encoder_mean = tf.layers.dense(
            h_encoders[-1],
            config.latent_dim,
            kernel_initializer=xav_init,
            name="encoder_mean",
            reuse=tf.AUTO_REUSE
        )
        encoder_log_var = tf.layers.dense(
            h_encoders[-1],
            config.latent_dim,
            kernel_initializer=xav_init,
            name="encoder_log_variance",
            reuse=tf.AUTO_REUSE
        )
    
    return encoder_mean, encoder_log_var

def decoder(Z):
    with tf.variable_scope("Decoder"):
        h_decoders = [
            tf.layers.dense(
                Z,
                config.decoder_hidden_size[0],
                activation=tf.nn.relu,
                kernel_initializer=xav_init,
                name="decoder_hidden_layer_0",
                reuse=tf.AUTO_REUSE
            )
        ]
        
        for index, size in enumerate(config.decoder_hidden_size[1:]):
            h_decoders.append(
                tf.layers.dense(
                    h_decoders[index],
                    size,
                    activation=tf.nn.relu,
                    kernel_initializer=xav_init,
                    name="decoder_hidden_layer_" + str(index + 1),
                    reuse=tf.AUTO_REUSE
                )
            )
        
        out_X = tf.layers.dense(
            h_decoders[-1],
            config.input_dim,
            kernel_initializer=xav_init,
            name="decoder_X",
            reuse=tf.AUTO_REUSE
        )
    
    return out_X, tf.nn.sigmoid(out_X)

def discriminator(U, n_layer=2, n_hidden=128):
    """ U:(n_samples, inp_data_dim, rank) """
    with tf.variable_scope("V", reuse=tf.AUTO_REUSE):
        # U = tf.reshape(U, [-1, latent_dim*rank])
        h = slim.repeat(U,n_layer,slim.fully_connected,n_hidden,activation_fn=tf.nn.leaky_relu,weights_regularizer=slim.l2_regularizer(0.1))
        h = slim.fully_connected(h,1,activation_fn=None,weights_regularizer=slim.l2_regularizer(0.1))
    return h

def get_disc_loss(p_samples, q_samples):
    """ formulates the loss function which optimises the discriminator such that the output of
        discriminator is log p/q """
    p_ratio = discriminator(p_samples)
    q_ratio = discriminator(q_samples)
    d_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p_ratio, labels=tf.ones_like(p_ratio)))
    d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q_ratio, labels=tf.zeros_like(q_ratio)))
    dloss_u = d_loss_d+d_loss_i
    return dloss_u

def get_forward_KL(p_samples):
    return tf.reduce_mean(discriminator(p_samples))

def sample_Z():
    global epsilon
    global encoder_mean, encoder_log_var
    
    return encoder_mean + tf.exp(encoder_log_var / 2) * epsilon

epoch_len = int(len(mnist.train.images) / config.batch_size)

X = tf.placeholder(tf.float32, [None, config.input_dim])
epsilon = tf.placeholder(tf.float32, [None, config.latent_dim])

################################################### VAE ###########################

encoder_mean, encoder_log_var = encoder(X)

z = sample_Z()

decoded_exp_X_mean, decoded_X_mean = decoder(z)

loss_recon = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=decoded_exp_X_mean), axis=1)
loss_rkl = 0.5*tf.reduce_sum(tf.exp(encoder_log_var) + encoder_mean**2. -1. - encoder_log_var, axis=1)


################################################## FAVI ############################

X_gen, _ = decoder(epsilon)
z_gen_mean, z_gen_log_var = encoder(X_gen)

loss_fkl = tf.reduce_mean(0.5*tf.reduce_sum(z_gen_log_var + tf.exp(-z_gen_log_var)*((z_gen_mean-epsilon)**2.), axis=1))
# loss_fkl = tf.reduce_mean(0.5*tf.reduce_sum(tf.exp(-z_gen_log_var)*((z_gen_mean-epsilon)**2.), axis=1))

################################################## Losses #########################

loss_vae = tf.reduce_mean(loss_rkl + loss_recon)

step_vae = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_vae)
step_fkl = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_fkl, var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder"))

##########################################

d_loss = get_disc_loss(decoded_X_mean, X) # discriminator loss Eq. 3.3 from AVB
loss_term = get_forward_KL(decoded_X_mean) # calculates E_p[log p/q]

step_d = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss, var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="V"))
step_forward = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_term, var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="Decoder"))

###########################################

saver = tf.train.Saver(var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="Decoder"))

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

def make_dirs():
    if not os.path.exists("Plots"):
        os.makedirs("Plots")
    if not os.path.exists("Plots/Samples"):
        os.makedirs("Plots/Samples")

def sample_plot(epoch):
    figure = np.zeros((280, 280))
    fig = plt.figure(figsize=(6, 6))
    
    for k in range(0, 10):
        for i in range(0, 10):
            eps = np.random.randn(1, config.latent_dim)
            decoded_image = sess.run(
                decoded_X_mean,
                feed_dict={
                    z: eps
                }
            ).reshape((28, 28)) * 255

            figure[k * 28 : (k + 1) * 28, i * 28 : (i + 1) * 28] = decoded_image
 
    plt.imshow(figure, cmap="Greys_r")

    plt.tight_layout()
    plt.savefig("Plots/Samples/" + str(epoch) + ".png")
    plt.close()


def latent_two(epoch):
	X_test = mnist.test.images
	y_test = mnist.test.labels
	for i in xrange(10):
		latent = sess.run(encoder_mean, feed_dict={X:X_test[i*1000:(i+1)*1000,:]})
		plt.scatter(latent[:,0], latent[:,1], c=y_test[i*1000:(i+1)*1000], cmap="tab20c", s=5)
	plt.tight_layout()
	plt.colorbar()
	plt.savefig("Plots/Samples/Latent_{}.png".format(str(epoch)))
	plt.close()

def tsne(epoch):
	pass

make_dirs()

def vae_routine():
	X_batch = mnist.train.next_batch(config.batch_size)[0]
	out = sess.run([loss_vae, step_vae], feed_dict={X:X_batch, epsilon: np.random.randn(config.batch_size, config.latent_dim)})
	return out[0]

def fkl_routine():
	X_batch = mnist.train.next_batch(config.batch_size)[0]
	out = sess.run([loss_fkl, step_fkl], feed_dict={X:X_batch, epsilon:np.random.randn(config.batch_size, config.latent_dim)})
	return out[0]

def disc_routine():
	# Train the discriminator to get the ratio accurately
	X_batch = mnist.train.next_batch(config.batch_size)[0]
	out = sess.run([d_loss, step_d], feed_dict={X:X_batch, epsilon: np.random.randn(config.batch_size, config.latent_dim)})
	return out[0]

def gen_routine():
	# Train the decoder according to gradient from discriminator
	X_batch = mnist.train.next_batch(config.batch_size)[0]
	out = sess.run([loss_term, step_forward], feed_dict={X:X_batch, epsilon: np.random.randn(config.batch_size, config.latent_dim)})
	return out[0]

# def vae_routine(epoch):
#     J = 0.0
#     for i in range(epoch_len):
#         X_batch = mnist.train.next_batch(config.batch_size)[0]
#         out = sess.run(
#             [loss_vae, step_vae],
#             feed_dict={
#                 X: X_batch,
#                 epsilon: np.random.randn(config.batch_size, config.latent_dim)
#             }
#         )
#         J += out[0] / epoch_len
    
#     print("Epoch %d: %.3f" % (epoch, J))
#     if epoch%10 == 0:
#         sample_plot(epoch)

# def fkl_routine(epoch):
#     J = 0.0
#     for i in range(epoch_len):
#         X_batch = mnist.train.next_batch(config.batch_size)[0]
#         out = sess.run(
#             [loss_fkl, step_fkl],
#             feed_dict={
#                 X: X_batch,
#                 epsilon: np.random.randn(config.batch_size, config.latent_dim)
#             }
#         )
#         J += out[0] / epoch_len
    
#     print("Epoch %d: %.3f" % (epoch, J))
#     if epoch%10 == 0:
#         sample_plot(epoch)

# def combined_routine(epoch):
#     J1 = J2 = 0.0
#     for i in range(epoch_len):
#         X_batch = mnist.train.next_batch(config.batch_size)[0]
#         out = sess.run(
#             [loss_vae, step_vae],
#             feed_dict={
#                 X: X_batch,
#                 epsilon: np.random.randn(config.batch_size, config.latent_dim)
#             }
#         )
#         J1 += out[0] / epoch_len

#         out = sess.run(
#             [loss_fkl, step_fkl],
#             feed_dict={
#                 X: X_batch,
#                 epsilon: np.random.randn(config.batch_size, config.latent_dim)
#             }
#         )
#         J2 += out[0] / epoch_len
    
#     print("Epoch %d: %.3f \t %.3f" % (epoch, J1, J2))
#     if epoch%10 == 0:
#         sample_plot(epoch)	

def old_code():
	for epoch in range(1,config.n_epochs+1):
		L_vae = 0.0
		L_fkl = 0.0
		for _ in xrange(epoch_len):
			L_vae += vae_routine()/epoch_len
			#L_fkl += fkl_routine()/epoch_len

		print "Epoch: %d \t %f \t %f" %(epoch, L_vae, L_fkl)
		sample_plot(epoch)
		# latent_two(epoch)

def paper_code():
	for epoch in range(1,config.n_epochs+1):
		L_vae = 0.0
		for _ in xrange(epoch_len):
			L_vae += vae_routine()/epoch_len
		print "Epoch: %d \t %f" %(epoch, L_vae)
		saver.save(sess, "./model.ckpt")
		sample_plot(epoch)
		latent_two(epoch)

	for epoch in range(1,config.n_epochs+1):
		L_fkl = 0.0
		for _ in xrange(epoch_len):
			L_fkl += fkl_routine()/epoch_len

		print "Epoch: %d \t %f" %(epoch, L_fkl)	
		sample_plot(epoch)

def alt_code_with_vae():
	for epoch in range(1,config.n_epochs+1):
		L_vae = 0.0
		L_fkl = 0.0
		for _ in xrange(epoch_len):
			L_vae += vae_routine()/epoch_len
			L_fkl += fkl_routine()/epoch_len
		print "Epoch: %d \t %f \t %f" %(epoch, L_vae, L_fkl)
		saver.save(sess, "./model.ckpt")
		sample_plot(epoch)
		latent_two(epoch)

def alt_code_with_disc():
	for epoch in range(1,config.n_epochs+1):
		L_fkl = 0.0
		L_gen = 0.0
		L_disc = 0.0
		for i in xrange(epoch_len):
			L_disc += disc_routine()/epoch_len
			if i%10 == 0:
				L_gen = gen_routine()/epoch_len
				L_fkl = fkl_routine()/epoch_len
		print "Epoch: %d \t %f \t %f \t %f" %(epoch, L_disc, L_gen, L_fkl)
		saver.save(sess, "./model.ckpt")
our_code()
		sample_plot(epoch)

paper_code()
