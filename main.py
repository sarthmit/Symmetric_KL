import os
import config
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from sklearn.manifold import TSNE

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

tf.logging.set_verbosity(tf.logging.ERROR)

ds = tf.contrib.distributions
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

################################################## Losses #########################

loss_vae = tf.reduce_mean(loss_rkl + loss_recon)

step_vae = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_vae)
step_fkl = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_fkl, var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder"))

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

make_dirs()

def vae_routine(epoch):
    J = 0.0
    for i in range(epoch_len):
        X_batch = mnist.train.next_batch(config.batch_size)[0]
        out = sess.run(
            [loss_vae, step_vae],
            feed_dict={
                X: X_batch,
                epsilon: np.random.randn(config.batch_size, config.latent_dim)
            }
        )
        J += out[0] / epoch_len
    
    print("Epoch %d: %.3f" % (epoch, J))
    if epoch%10 == 0:
        sample_plot(epoch)

def fkl_routine(epoch):
    J = 0.0
    for i in range(epoch_len):
        X_batch = mnist.train.next_batch(config.batch_size)[0]
        out = sess.run(
            [loss_fkl, step_fkl],
            feed_dict={
                X: X_batch,
                epsilon: np.random.randn(config.batch_size, config.latent_dim)
            }
        )
        J += out[0] / epoch_len
    
    print("Epoch %d: %.3f" % (epoch, J))
    if epoch%10 == 0:
        sample_plot(epoch)

def combined_routine(epoch):
    J1 = J2 = 0.0
    for i in range(epoch_len):
        X_batch = mnist.train.next_batch(config.batch_size)[0]
        out = sess.run(
            [loss_vae, step_vae],
            feed_dict={
                X: X_batch,
                epsilon: np.random.randn(config.batch_size, config.latent_dim)
            }
        )
        J1 += out[0] / epoch_len

        out = sess.run(
            [loss_fkl, step_fkl],
            feed_dict={
                X: X_batch,
                epsilon: np.random.randn(config.batch_size, config.latent_dim)
            }
        )
        J2 += out[0] / epoch_len
    
    print("Epoch %d: %.3f \t %.3f" % (epoch, J1, J2))
    if epoch%10 == 0:
        sample_plot(epoch)	

for epoch in range(1,config.n_epochs+1):
    combined_routine(epoch)
    # vae_routine(epoch)
    # fkl_routine(epoch)

for epoch in range(1,11):
    fkl_routine(epoch)