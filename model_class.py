import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import argparse
import pickle
import numpy as np

## hyperparameters: (originally from Bowman et.al., might change in the future)
lstm_dim = 191
z_dim = 13
embedding_dim = 353

#train_data_path = "data/yelp_train.pkl"
#dev_data_path = "data/yelp_dev.pkl"
#word_inds_path = "data/yelp_word_inds.pkl"
train_data_path = "data/yelp_train_25k.pkl"
dev_data_path = "data/yelp_dev_25k.pkl"
word_inds_path = "data/yelp_word_inds_25k.pkl"


# remove this part once we have preprocessed data
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
tokenizer = info.features['text'].encoder
vocab_size = tokenizer.vocab_size
BUFFER_SIZE = 10000
BATCH_SIZE = 64
max_len = 25
#start_ind = 19#np.ndarray([19], dtype=np.float32)
start_ind = np.array([19], dtype=np.float32)

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

def to_inds(sent):
    inds = tokenizer.encode(sent)
    pad_len = max_len - len(inds)
    inds += [0] * pad_len
    inds = np.array(inds).reshape((1,max_len))
    return inds

def to_words(inds):
    inds = [ind.numpy()[0] for ind in inds]
    words = tokenizer.decode(inds)
    return words

class SentVae(tf.keras.Model):

    def __init__(self):
        super(SentVae, self).__init__()
        self.emb_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, name='emb')
        #self.inf_lstm = tf.keras.layers.LSTM(lstm_dim, name='inf_out')
        self.inf_lstm = tf.keras.layers.RNN(cell=tf.keras.layers.LSTMCell(lstm_dim), name='inf_out')
        self.mu_layer = tf.keras.layers.Dense(z_dim, name='mu')
        self.sigma_layer = tf.keras.layers.Dense(z_dim, name='sigma')
        self.mu = [0] * z_dim
        self.sigma = [0] * z_dim

        self.init_state_layer = tf.keras.layers.Dense(lstm_dim, input_shape=(z_dim,), name='init_state')
        self.gen_lstm_layer = tf.keras.layers.LSTMCell(lstm_dim, name='gen_lstm_layer')
        self.state_to_inds = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False):
        inf_emb = self.emb_layer(inputs)
        inf_out = self.inf_lstm(inf_emb)

        mu = self.mu_layer(inf_out)
        self.mu = mu
        sigma = self.sigma_layer(inf_out)
        self.sigma = sigma

        dist = tfp.distributions.Normal(loc=0., scale=1.)
        epsilon = dist.sample(sample_shape=(z_dim,))

        z = tf.add(mu, tf.math.multiply(epsilon, sigma))
        init_state = self.init_state_layer(z)
        h, c = init_state, init_state

        if training: # teacher forcing
            #inp_inds = tf.transpose(inputs, perm=[1,0,2])
            inp_inds = tf.transpose(inputs, perm=[1,0])
            inds_out = []
            for ind in tf.unstack(inp_inds):
                #print(self.emb_layer(ind).shape)
                #emb = tf.expand_dims(self.emb_layer(ind), axis=0)
                emb = self.emb_layer(ind)
                #emb = tf.reshape(self.emb_layer(ind), (1,embedding_dim))
                #print(emb.shape)
                #print(h.shape)
                #print(c.shape)
                curr_out, (h,c) = self.gen_lstm_layer(emb, states=[h,c])
                ind = self.state_to_inds(curr_out)
                inds_out.append(ind)

            #output = (tf.stack(inds_out, axis=1), mu, sigma)
            output = tf.stack(inds_out, axis=1)
        else:
            #emb = tf.reshape(self.emb_layer(start_ind), (1,embedding_dim))
            emb = self.emb_layer(start_ind)
            inds_out = []

            for _ in range(max_len):
                curr_out, (h,c) = self.gen_lstm_layer(emb, states=[h,c])
                new_ind_oh = self.state_to_inds(curr_out)
                new_ind = tf.argmax(new_ind_oh, axis=1)
                #emb = tf.reshape(self.emb_layer(new_ind), (1,embedding_dim))
                #emb = tf.expand_dims(self.emb_layer(new_ind), axis=0)
                emb = self.emb_layer(new_ind)
                inds_out.append(new_ind_oh)
            #output = (tf.stack(inds_out, axis=1), mu, sigma)
            output = tf.stack(inds_out, axis=1)

        #output = tf.stack(inds_out, axis=1)
        return output

    def inference(self, sent):
        inds = to_inds(sent)

        #for ind in inds:
        #    emb = self.emb_layer(ind)
        embedded = self.emb_layer(inds)
        inf_out = self.inf_lstm(embedded)

        mu = self.mu_layer(inf_out)
        sigma = self.sigma_layer(inf_out)

        return mu, sigma

    def sample_z(self, mu, sigma):
        dist = tfp.distributions.Normal(loc=0., scale=1.)
        epsilon = dist.sample(sample_shape=(z_dim,))

        z = tf.add(mu, tf.math.multiply(epsilon, sigma))

        return z

    def generate(self, z):
        init_state = self.init_state_layer(z)
        h, c = init_state, init_state
            
        emb = self.emb_layer(start_ind)
        inds_out = []

        for _ in range(max_len):
            curr_out, (h,c) = self.gen_lstm_layer(emb, states=[h,c])
            new_ind_oh = self.state_to_inds(curr_out)
            new_ind = tf.argmax(new_ind_oh, axis=1)
            emb = self.emb_layer(new_ind)
            inds_out.append(new_ind)

        sent = to_words(inds_out)

        return sent
        
