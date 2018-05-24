import argparse
import os
import pickle
from random import shuffle

import numpy as np

import multitask_lang_model
import multitask_vision_model
from utils import MyModelCheckpoint

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
    preprocessed_dataset_path = "lang_dataset/"
    embeddings_filename = "/mnt/povobackup/clic/sandro.pezzelle/corpus-and-vectors/GoogleNews-vectors-negative300.txt"
    vision_weights_filename = "/mnt/povobackup/clic/sandro.pezzelle/model_weights_correct/multitask-prop-weights-sgdlr05-correct.hdf5"
    lang_weights_filename = "best_models/vision_to_lang_model-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dataset_path", type=str, default=preprocessed_dataset_path)
    parser.add_argument("--embeddings_filename", type=str, default=embeddings_filename)
    parser.add_argument("--vision_weights_filename", type=str, default=vision_weights_filename)
    parser.add_argument("--lang_weights_filename", type=str, default=lang_weights_filename)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    index_filename = os.path.join(args.preprocessed_dataset_path, "index.pkl")
    print("Loading filename: {}".format(index_filename))
    with open(index_filename, mode="rb") as in_file:
        index = pickle.load(in_file)
        token2id = index["token2id"]
        id2token = index["id2token"]

    train_filename = os.path.join(args.preprocessed_dataset_path, "train.pkl")
    print("Loading filename: {}".format(train_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "train.pkl"), mode="rb") as in_file:
        train = pickle.load(in_file)
        dataset_tr = train["dataset_tr"]
        tr_m_out = train["tr_m_out"]
        tr_q_out = train["tr_q_out"]
        tr_r_out = train["tr_r_out"]

    test_filename = os.path.join(args.preprocessed_dataset_path, "test.pkl")
    print("Loading filename: {}".format(test_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "test.pkl"), mode="rb") as in_file:
        test = pickle.load(in_file)
        dataset_t = test["dataset_t"]
        t_m_out = test["t_m_out"]
        t_q_out = test["t_q_out"]
        t_r_out = test["t_r_out"]

    valid_filename = os.path.join(args.preprocessed_dataset_path, "valid.pkl")
    print("Loading filename: {}".format(valid_filename))
    with open(os.path.join(args.preprocessed_dataset_path, "valid.pkl"), mode="rb") as in_file:
        valid = pickle.load(in_file)
        dataset_v = valid["dataset_v"]
        v_m_out = valid["v_m_out"]
        v_q_out = valid["v_q_out"]
        v_r_out = valid["v_r_out"]

    print("Loading filename: {}".format(args.embeddings_filename))
    embeddings_index = {}
    with open(args.embeddings_filename) as in_file:
        for line in in_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(token2id) + 1, 300))
    for word, i in token2id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("Training model...")
    vision_model = multitask_vision_model.MultitaskVisionModel().build()
    vision_model.load_weights(args.vision_weights_filename)
    lang_model = multitask_lang_model.MultitaskLangModel(embedding_matrix, token2id).build()

    for lvis, llang in zip(vision_model.layers[3:], lang_model.layers[7:]):
        llang.set_weights(lvis.get_weights())
        llang.trainable = False

    checkpoint = MyModelCheckpoint(args.lang_weights_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    hist = lang_model.fit(
        dataset_tr,
        [tr_m_out, tr_q_out, tr_r_out],
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        validation_data=(dataset_v, [v_m_out, v_q_out, v_r_out]),
        callbacks=[checkpoint]
    )

    print("Evaluating model...")
    best_model = multitask_lang_model.MultitaskLangModel(embedding_matrix, token2id).build()
    best_model.load_weights(checkpoint.last_saved_filename)
    scores = best_model.evaluate(dataset_t, [t_m_out, t_q_out, t_r_out], batch_size=args.batch_size)
    print("%s: %.4f%%" % (best_model.metrics_names[1], scores[1] * 100))
