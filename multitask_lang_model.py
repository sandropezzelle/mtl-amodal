from keras import backend as K
from keras.layers import Embedding
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Reshape
from keras.layers.core import Lambda
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.regularizers import l2


class MultitaskLangModel:
    def __init__(self, embedding_matrix, token2id, emb_dim=300, input_shape=(25, 50), act_f='relu', dropout=0.5,
                 l2_reg=1e-8, batch_size=32, multitask_vision_model=None):
        """
        initialization of hyperparameters
        """
        self._embedding_matrix = embedding_matrix
        self._token2id = token2id
        self._emb_dim = emb_dim
        self._input_shape = input_shape
        self._act_f = act_f
        self._dropout = dropout
        self._l2_reg = l2_reg
        self._batch_size = batch_size
        self._more_classes = 3
        self._prop_classes = 17
        self._q_classes = 9
        self._multitask_vision_model = multitask_vision_model

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
        emb_mod = Embedding(len(self._token2id) + 1, self._emb_dim, weights=[self._embedding_matrix], trainable=False)
        sum_dim1 = Lambda(lambda xin: K.sum(xin, axis=2))

        inp_res = Reshape((25 * 50,))(inp)
        emb_out = emb_mod(inp_res)
        res_emb = Reshape((25, 50, self._emb_dim))(emb_out)
        res_sum_dim1 = sum_dim1(res_emb)

        td_dense0 = TimeDistributed(Dense(2048, W_regularizer=l2(self._l2_reg), activation=self._act_f))
        drop_td_dense0 = Dropout(self._dropout)
        l_td_dense0 = td_dense0(res_sum_dim1)
        drop_l_td_dense0 = drop_td_dense0(l_td_dense0)

        td_dense = TimeDistributed(Dense(1024, W_regularizer=l2(self._l2_reg), activation=self._act_f))
        drop_td_dense = Dropout(self._dropout)
        l_td_dense = td_dense(drop_l_td_dense0)
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

        model = Model(input=inp, output=[out_more, out_quant, out_prop])
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        if self._multitask_vision_model:
            for lvis, llang in zip(self._multitask_vision_model.layers[3:], model.layers[7:]):
                print(lvis, llang)
                llang.set_weights(lvis.get_weights())
                llang.trainable = False

        model = Model(input=inp, output=[out_more, out_quant, out_prop])
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        return model
