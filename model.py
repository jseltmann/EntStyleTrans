import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

## hyperparameters: (originally from Bowman et.al., might change in the future)
lstm_dim = 191
z_dim = 13
embedding_dim = 353


# remove this part once we have preprocessed data
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
tokenizer = info.features['text'].encoder
vocab_size = tokenizer.vocab_size
BUFFER_SIZE = 10000
BATCH_SIZE = 64
max_len = 25

train_dataset = train_dataset.map(lambda x,y : (x,x))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.filter(lambda x,y: len(x) <= max_len)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, ((max_len,),(max_len,)))


test_dataset = test_dataset.map(lambda x,y : (x,x))
test_dataset = test_dataset.filter(lambda x,y: len(x) <= max_len)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, ((max_len,),(max_len,)))

emb_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, name='emb')

inputs = tf.keras.layers.Input(shape=(max_len,), name='inputs')
inf_emb = emb_layer(inputs)
inf_out = tf.keras.layers.LSTM(lstm_dim, name='inf_out')(inf_emb)

mu = tf.keras.layers.Dense(z_dim, name='mu')(inf_out)
sigma = tf.keras.layers.Dense(z_dim, name='sigma')(inf_out)



dist = tfp.distributions.Normal(loc=0., scale=1.)
epsilon = dist.sample(sample_shape=(z_dim,))

z = mu + epsilon * sigma
inf_net = tf.keras.Model(inputs, z)
#inf_net.summary()


# build generator net
gen_inputs = tf.keras.layers.Input(shape=(max_len,), name='gen_inputs')
gen_z = tf.keras.layers.Input(shape=(z_dim,), name='gen_z')
gen_emb = emb_layer(gen_inputs)

init_state_layer = tf.keras.layers.Dense(lstm_dim, input_shape=(z_dim,), name='init_state')
init_state = init_state_layer(gen_z)


gen_lstm_layer = tf.keras.layers.LSTM(lstm_dim,
                                      return_sequences=True,
                                      name='gen_lstm_layer')

gen_out = gen_lstm_layer(gen_emb, initial_state=[init_state, init_state])
gen_out_list = tf.unstack(gen_out, axis=1)


state_to_inds = tf.keras.layers.Dense(vocab_size)
inds_out = []

for state in gen_out_list:
    inds_vector = state_to_inds(state)
    #inds = tf.cast(tf.argmax(inds_vector, axis=1), tf.float32)
    #inds_out.append(inds)
    inds_out.append(inds_vector)

    




#gen_net = tf.keras.Model([gen_inputs, gen_z], gen_out)
gen_net = tf.keras.Model([gen_inputs, gen_z], inds_out)


outputs = gen_net([inputs, inf_net(inputs)])
outputs = tf.stack(outputs, axis=1)

vae = tf.keras.Model(inputs, outputs)


def vae_loss(true, pred):
    true = tf.cast(true, tf.int32)
    true = tf.one_hot(true, vocab_size)
    print(true[0].shape)
    print(pred[0].shape)
    print("---------------------")
    reconstruction_loss = tf.keras.losses.MeanSquaredError()(true, pred)
    #reconstruction_loss = tf.keras.losses.MeanAbsoluteError()(true, pred)
    reconstruction_loss *= max_len

    #return reconstruction_loss
    kl_loss = 1 + sigma - tf.square(mu) - tf.exp(sigma)
    kl_loss = -0.5 * tf.reduce_sum(kl_loss, 1)

    loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    return loss



vae.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=vae_loss)#,
            #metrics=['accuracy'])


history = vae.fit(train_dataset,
                  epochs=1,
                  validation_data=test_dataset)

