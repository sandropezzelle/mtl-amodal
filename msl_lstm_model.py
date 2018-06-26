import numpy as np
from keras import optimizers
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.layers import Reshape
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.regularizers import l2


class MSLLSTMModel:
    def __init__(self, embedding_matrix, token2id, emb_dim=300, input_shape=(25, 50), act_f='relu', dropout=0.5, l2_reg=1e-8, batch_size=100):
        """
        initialization of hyperparameters
        """
        self._embedding_matrix = embedding_matrix
        self._token2id = token2id
        self._emb_dim = emb_dim
        self._batch_size = batch_size
        self._input_shape = input_shape
        self._act_f = act_f
        self._dropout = dropout
        self._l2_reg = l2_reg
        self._more_classes = 3

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
        inp = Input(self._input_shape, name='lang_input')
        emb_mod = Embedding(len(self._token2id) + 1, self._emb_dim, weights=[self._embedding_matrix], trainable=True)
        lstm_mod = LSTM(300, activation=self._act_f)

        inp_res = Reshape((25 * 50,))(inp)
        emb_out = emb_mod(inp_res)
        res_emb = Reshape((25, 50, self._emb_dim))(emb_out)
        td_lstm = TimeDistributed(lstm_mod)
        res_td_lstm = td_lstm(res_emb)

        more_flat = Flatten()(res_td_lstm)
        hidden_more = Dense(2048, W_regularizer=l2(self._l2_reg), activation=self._act_f, name='msl')(more_flat)
        drop_hidden_more = Dropout(self._dropout)(hidden_more)
        out_more = Dense(self._more_classes, activation='softmax', name='pred1')(drop_hidden_more)

        model = Model(inputs=inp, outputs=out_more)
        sgd = optimizers.SGD(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        return model
