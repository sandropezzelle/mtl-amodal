import sys
from random import shuffle

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

import model

import msl_lstm_model
import utils

data_path = ''
ratios = {}
r_dict = {}


def load_people(target_index, non_target_index):
    target = data_path + '1900_133.txt'
    non_target = data_path + '1700_133.txt'

    with open(target, 'r') as targx:
        reader = [line.split('\t')[3] for line in targx]

    with open(non_target, 'r') as ntargx:
        nreader = [line.split('\t')[3] for line in ntargx]

    people = []

    for idx in target_index:
        if idx is not '':
            i = int(idx)
            myperson = reader[i].split()[0:50]
            people.append(myperson)

    for idx in non_target_index:
        if idx is not '':
            j = int(idx)
            myperson = nreader[j].split()[0:50]
            people.append(myperson)

    pad = 25 - int(len(people))

    for i in range(pad):
        people.append([])

    shuffle(people)

    return people


def read_data(data_path):
    """
    it reads train/validation/test files
    """
    tr = data_path + 'train_shuf.txt'
    v = data_path + 'val_shuf.txt'
    tst = data_path + 'test_shuf.txt'
    return tr, v, tst


def load_data(split):
    """
    for a given list of paths, it reads the image,
    prepares it for input and it calculates the target value
    """
    with open(split, 'r') as splitfile:
        reader = [line.split() for line in splitfile]
        size = int(len(reader))
        inp = [[] for _ in range(size)]
        m_out = np.zeros((size, 3))
        q_out = np.zeros((size, 9))
        r_out = np.zeros((size, 17))
        count = 0

    with open(split, 'r') as splitfile:
        for n, line in enumerate(splitfile):
            line2 = line.split("', ")[2][1:]
            target = line2.split("], [")[0]
            target_arr = target.split(", ")
            non_target = line2.split("], [")[1].strip("])']\n")
            non_target_arr = non_target.split(", ")
            cat = line[2:4]

            for i in range(9):
                q_out[count][i] = ratios[cat][str(i)]

            r_out[count][r_dict[cat]] = 1.0

            if cat[1] == 'Y' or cat[0] == 'X':
                if cat[1] == 'Y':
                    ratio_val = 0.0
                else:
                    ratio_val = 1.0
            else:
                ratio_val = float(cat[0]) / (float(cat[1]) + float(cat[0]))

            if ratio_val < 0.5:
                m_out[count][0] = 1.0

            if ratio_val == 0.5:
                m_out[count][1] = 1.0

            if ratio_val > 0.5:
                m_out[count][2] = 1.0

            inpl = load_people(target_arr, non_target_arr)
            inp[count].append(inpl)
            count += 1

        return inp, m_out, q_out, r_out


def create_ratio_dict(ratios):
    count = 0
    r = sorted(ratios.keys())
    for i in range(len(r)):
        r_dict[r[i]] = count
        count += 1


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
    weights = "msl_lstm_model/best-weight-model.hdf5"

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

    model = msl_lstm_model.MSLLSTMModel(embeddings, token2id).build()

    checkpoint = ModelCheckpoint(weights, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    hist = model.fit(
        dataset_tr,
        tr_m_out,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(dataset_v, v_m_out),
        callbacks=[checkpoint]
    )

    best_model = model.MultitaskLangModel(embeddings, token2id).build()
    best_model.load_weights(weights)
    print(best_model.evaluate(dataset_t, t_m_out, batch_size=batch_size))
