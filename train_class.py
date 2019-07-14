import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import argparse
import pickle

from model_class import SentVae

train_data_path = "data/yelp_train.pkl"
dev_data_path = "data/yelp_dev.pkl"
word_inds_path = "data/yelp_word_inds.pkl"

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
BATCH_SIZE = 8#64
max_len = 25

#with open(train_data_path, "rb") as tdf:
#    train_data = pickle.load(tdf)
#with open(dev_data_path, "rb") as ddf:
#    dev_data = pickle.load(ddf)
#with open(word_inds_path, "rb") as wif:
#    word_inds = pickle.load(wif)
#    word2num = word_inds["word2num"]
#    num2word = word_inds["num2word"]
#
#vocab_size = len(word2num)


def to_one_hot(x):
    true = tf.cast(x, tf.int32)
    true = tf.one_hot(true, vocab_size)
    return true

#train_data = [to_one_hot(x) for x in train_data]
#dev_data = [to_one_hot(x) for x in dev_data]

train_dataset = train_dataset.map(lambda x,y : (x[:max_len],x[:max_len]))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, ((max_len,),(max_len,)))
train_dataset = train_dataset.map(lambda x,y : (x,to_one_hot(x)))

test_dataset = test_dataset.map(lambda x,y : (x[:max_len],x[:max_len]))
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, ((max_len,),(max_len,)))
test_dataset = test_dataset.map(lambda x,y : (x,to_one_hot(x)))


vae = SentVae()


def vae_loss(true, pred):
    reconstruction_loss = tf.keras.losses.MeanSquaredError()(true, pred)
    reconstruction_loss *= max_len

    #calculate mu and sigma
    inf_emb = vae.emb_layer(true)[0]
    inf_out = vae.inf_lstm(inf_emb)
    mu = vae.mu_layer(inf_out)
    sigma = vae.sigma_layer(inf_out)

    kl_loss = 1 + sigma - tf.square(mu) - tf.exp(sigma)
    kl_loss = -0.5 * tf.reduce_sum(kl_loss, 1)

    loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weightsfn", default="model_weights")
    parser.add_argument("--logdir", default="logs")

    args = parser.parse_args()
    weights_filename = args.weightsfn
    model_filename = args.modelfn
    logdir = args.logdir

    
    vae.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=vae_loss)

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    history = vae.fit(train_dataset,
                      epochs=1,
                      validation_data=test_dataset)#,
                      #callbacks=[tensorboard_callback])
    #history = va.fit(train_data, train_data,
    #                  epochs=1,
    #                  validation_data=(test_dataset,test_dataset))

    vae.save_weights(weights_filename)
