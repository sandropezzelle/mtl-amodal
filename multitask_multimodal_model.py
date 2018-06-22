from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf


class MultitaskMultimodalModel:
    def __init__(self, input_shape=(203, 203, 3), act_f='relu', dropout=0.5, l2_reg=1e-8, batch_size=32):
        self._input_shape = input_shape
        self._act_f = act_f
        self._dropout = dropout
        self._l2_reg = l2_reg
        self._batch_size = batch_size
        self._more_classes = 3
        self._prop_classes = 17
        self._q_classes = 9

    def build(self):
        # Language input

        language_input = Input(self._input_shape, name='lang_input')
        emb_mod = Embedding(len(self._token2id) + 1, self._emb_dim, weights=[self._embedding_matrix], trainable=False)
        lstm_mod = LSTM(2048, activation=self._act_f)
        inp_res = Reshape((25 * 50,))(language_input)
        emb_out = emb_mod(inp_res)
        res_emb = Reshape((25, 50, self._emb_dim))(emb_out)
        td_lstm = TimeDistributed(lstm_mod)
        res_td_lstm = td_lstm(res_emb)

        td_dense_lang = TimeDistributed(Dense(2048, W_regularizer=l2(self._l2_reg), activation=self._act_f))
        drop_td_dense_lang = Dropout(self._dropout)
        l_td_dense_lang = td_dense_lang(res_td_lstm)
        drop_l_td_dense_lang = drop_td_dense_lang(l_td_dense_lang)

        # Visual input

        model_inception = InceptionV3(weights=None, include_top=False)
        visual_input = Input(self._input_shape, name='more_input')
        out_inc = model_inception(visual_input)
        out_res = Reshape((25, 2048))(out_inc)

        td_dense_visual = TimeDistributed(Dense(2048, W_regularizer=l2(self._l2_reg), activation=self._act_f))
        drop_td_dense_visual = Dropout(self._dropout)
        l_td_dense_visual = td_dense_visual(out_res)
        drop_l_td_dense_visual = drop_td_dense_visual(l_td_dense_visual)

        # Shared layers

        def shared_model(x):
            td_dense = TimeDistributed(Dense(1024, W_regularizer=l2(self._l2_reg), activation=self._act_f))
            drop_td_dense = Dropout(self._dropout)
            l_td_dense = td_dense(x)
            drop_l_td_dense = drop_td_dense(l_td_dense)

            td_dense2 = TimeDistributed(Dense(512, W_regularizer=l2(self._l2_reg), activation=self._act_f))
            drop_td_dense2 = Dropout(self._dropout)
            l_td_dense2 = td_dense2(drop_l_td_dense)
            drop_l_td_dense2 = drop_td_dense2(l_td_dense2)

            td_dense3 = TimeDistributed(Dense(256, W_regularizer=l2(self._l2_reg), activation=self._act_f))
            drop_td_dense3 = Dropout(self._dropout)
            l_td_dense3 = td_dense3(drop_l_td_dense2)
            drop_l_td_dense3 = drop_td_dense3(l_td_dense3)

            td_dense4 = TimeDistributed(Dense(128, W_regularizer=l2(self._l2_reg), activation=self._act_f))
            drop_td_dense4 = Dropout(self._dropout)
            l_td_dense4 = td_dense4(drop_l_td_dense3)
            drop_l_td_dense4 = drop_td_dense4(l_td_dense4)

            td_dense5 = TimeDistributed(Dense(64, W_regularizer=l2(self._l2_reg), activation=self._act_f))
            drop_td_dense5 = Dropout(self._dropout)
            l_td_dense5 = td_dense5(drop_l_td_dense4)
            drop_l_td_dense5 = drop_td_dense5(l_td_dense5)

            td_dense6 = TimeDistributed(Dense(32, W_regularizer=l2(self._l2_reg), activation=self._act_f))
            drop_td_dense6 = Dropout(self._dropout)
            l_td_dense6 = td_dense6(drop_l_td_dense5)
            drop_l_td_dense6 = drop_td_dense6(l_td_dense6)

            more_flat = Flatten()(drop_l_td_dense2)
            hidden_more = Dense(512, W_regularizer=l2(self._l2_reg), activation=self._act_f, name='msl')(more_flat)
            drop_hidden_more = Dropout(self._dropout)(hidden_more)
            out_more = Dense(self._more_classes, activation='softmax', name='pred1')(drop_hidden_more)

            quant_flat = Flatten()(drop_l_td_dense4)
            hidden_quant = Dense(128, W_regularizer=l2(self._l2_reg), activation=self._act_f, name='quant')(quant_flat)
            drop_hidden_quant = Dropout(self._dropout)(hidden_quant)
            out_quant = Dense(self._q_classes, activation='softmax', name='pred2')(drop_hidden_quant)

            prop_flat = Flatten()(drop_l_td_dense6)
            hidden_prop = Dense(32, W_regularizer=l2(self._l2_reg), activation=self._act_f, name='prop')(prop_flat)
            drop_hidden_prop = Dropout(self._dropout)(hidden_prop)
            out_prop = Dense(self._prop_classes, activation='softmax', name='pred3')(drop_hidden_prop)

            return out_more, out_quant, out_prop

        sgd = optimizers.SGD(lr=0.001)

        language_model = Model(inputs=language_input, outputs=shared_model(language_input))
        language_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        visual_model = Model(inputs=visual_input, outputs=shared_model(visual_input))
        visual_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return language_model
