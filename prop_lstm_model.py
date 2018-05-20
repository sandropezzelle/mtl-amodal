import numpy as np
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.layers import Reshape
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.regularizers import l2


class PropLSTMModel:
    def __init__(self, embeddings, token2id, emb_dim=300, input_shape=(25, 50), act_f='relu', dropout=0.5, l2_reg=1e-8, batch_size=100):
        """
        initialization of hyperparameters
        """
        self._embeddings = embeddings
        self._token2id = token2id
        self._emb_dim = emb_dim
        self._batch_size = batch_size
        self._input_shape = input_shape
        self._act_f = act_f
        self._dropout = dropout
        self._l2_reg = l2_reg
        self._more_classes = 3
        self._prop_classes = 17
        self._q_classes = 9

    def build(self):
        """
        loads the inception network,
        contains the fully connected layers,
        for each task, it predicts the class
        it returns the prediction for 3 tasks

        model_inception = InceptionV3(weights = None, include_top = False)
        inp = Input(self.input_shape, name = 'more_input')

        out_inc = model_inception(inp)
        out_res = Reshape((25,2048))(out_inc)
        """
        embeddings_index = {}
        with open(self._embeddings) as in_file:
            for line in in_file:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((len(self._token2id) + 1, self._emb_dim))
        for word, i in self._token2id.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        inp = Input(self._input_shape, name='lang_input')
        emb_mod = Embedding(len(self._token2id) + 1, self._emb_dim, weights=[embedding_matrix], trainable=False)
        lstm_mod = LSTM(300, activation=self._act_f)

        inp_res = Reshape((25 * 50,))(inp)
        emb_out = emb_mod(inp_res)
        res_emb = Reshape((25, 50, self._emb_dim))(emb_out)
        td_lstm = TimeDistributed(lstm_mod)
        res_td_lstm = td_lstm(res_emb)

        prop_flat = Flatten()(res_td_lstm)
        hidden_prop = Dense(2048, W_regularizer=l2(self._l2_reg), activation=self._act_f, name='msl')(prop_flat)
        drop_hidden_prop = Dropout(self._dropout)(hidden_prop)
        out_prop = Dense(self._prop_classes, activation='softmax', name='pred2')(drop_hidden_prop)

        model = Model(input=inp, output=out_prop)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model
