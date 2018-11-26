input_dim = 784
latent_dim = 5

n_epochs = 50
batch_size = 100

regularizer = 1

encoder_hidden_size = [1024,512,256,128]
decoder_hidden_size = [128,256,512,1024]

adam_decay_steps = 10
adam_decay_rate = 0.9
adam_learning_rate = 0.002
adam_epsilon = 1e-04
