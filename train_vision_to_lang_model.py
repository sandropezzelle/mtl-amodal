import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

import multitask_lang_model
import multitask_vision_model
import utils
from dataset import read_data, create_ratio_dict, load_data

if __name__ == '__main__':
    """
    it reads the parameters,
    initializes the hyperparameters,
    preprocesses the input,
    trains the model
    """
    repository_path = sys.argv[1]
    data_path = sys.argv[2]
    tr, v, tst = read_data(data_path)

    ratios = utils.read_qprobs(repository_path)
    create_ratio_dict(ratios)

    tr_inp, tr_m_out, tr_q_out, tr_r_out = load_data(tr)
    v_inp, v_m_out, v_q_out, v_r_out = load_data(v)
    t_inp, t_m_out, t_q_out, t_r_out = load_data(tst)

    token2id = {}
    id2token = {}
    dataset_tr, dataset_v, dataset_t = [], [], []

    num_epochs = 100
    learning_rate = 0.001
    batch_size = 32
    embeddings = '/mnt/povobackup/clic/sandro.pezzelle/corpus-and-vectors/GoogleNews-vectors-negative300.txt'
    vision_weights = "/mnt/povobackup/clic/sandro.pezzelle/model_weights_correct/multitask-prop-weights-sgdlr05-correct.hdf5"
    lang_weights = "vision_to_lang_model/best-weight-model.hdf5"

    for n, dtp in enumerate(tr_inp):
        pad = np.zeros((25, 50))
        for i in dtp:
            authors = []
            for nn, el in enumerate(i):
                author = []
                for token in el:
                    if not token in token2id:
                        i = len(token2id) + 1
                        token2id[token] = i
                        id2token[i] = token
                    author.append(token2id[token])
                authors.append(author)
            authors = pad_sequences(authors, padding='post')
            dataset_tr.append(authors)
    dataset_tr = np.array(dataset_tr)

    for n, dtp in enumerate(v_inp):
        pad = np.zeros((25, 50))
        for i in dtp:
            authors = []
            for nn, el in enumerate(i):
                author = []
                for token in el:
                    if not token in token2id:
                        i = len(token2id) + 1
                        token2id[token] = i
                        id2token[i] = token
                    author.append(token2id[token])
                authors.append(author)
            authors = pad_sequences(authors, padding='post')
            dataset_v.append(authors)
    dataset_v = np.array(dataset_v)

    for n, dtp in enumerate(t_inp):
        pad = np.zeros((25, 50))
        for i in dtp:
            authors = []
            for nn, el in enumerate(i):
                author = []
                for token in el:
                    if not token in token2id:
                        i = len(token2id) + 1
                        token2id[token] = i
                        id2token[i] = token
                    author.append(token2id[token])
                authors.append(author)
            authors = pad_sequences(authors, padding='post')
            dataset_t.append(authors)
    dataset_t = np.array(dataset_t)

    multitask_vision_model = multitask_vision_model.MultitaskVisionModel().build()
    multitask_vision_model.load_weights(vision_weights)
    multitask_lang_model = multitask_lang_model.MultitaskLangModel(embeddings, token2id).build()

    for lvis, llang in zip(multitask_vision_model.layers[3:], multitask_lang_model.layers[7:]):
        llang.set_weights(lvis.get_weights())
        llang.trainable = False

    checkpoint = ModelCheckpoint(lang_weights, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    hist = multitask_lang_model.fit(
        dataset_tr,
        [tr_m_out, tr_q_out, tr_r_out],
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(dataset_v, [v_m_out, v_q_out, v_r_out]),
        callbacks=[checkpoint]
    )

    multitask_lang_model = multitask_lang_model.MultitaskLangModel(embeddings, token2id).build()
    multitask_lang_model.load_weights(lang_weights)
    print(multitask_lang_model.evaluate(dataset_t, [t_m_out, t_q_out, t_r_out], batch_size=batch_size))
