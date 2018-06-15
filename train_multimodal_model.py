import argparse
import os
import pickle

import numpy as np

import utils

full_images = {}
ratios = {}
r_dict = {}
data_path = ''


def read_ind_files(path):
    """
    reads the indices of the images for a given file
    """
    fin = open(path, 'r')
    links = []
    for line in fin:
        links.append(data_path + line.strip())
    return links


def read_indices(repository_path):
    """
    it goes through train/test/validation files
    """
    path = repository_path + '/code/data_split/'
    tr = path + 'train_ids.txt'
    v = path + 'valid_ids.txt'
    t = path + 'test_ids.txt'
    tr_links = read_ind_files(tr)
    v_links = read_ind_files(v)
    t_links = read_ind_files(t)
    return tr_links, v_links, t_links


def read_images(links, size):
    """
    for a given list of paths, it reads the image,
    prepares it for input and it calculates the target value
    """
    dim = 203
    inp = np.zeros((size, dim, dim, 3))
    m_out = np.zeros((size, 3))
    q_out = np.zeros((size, 9))
    r_out = np.zeros((size, 17))
    count = 0
    for link in links[:size]:
        res_img = utils.load_image(link, dim)
        inp[count] = res_img
        cat = link.strip().split('/')[-2][-2:]
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
        count += 1
        if count % 100 == 0:
            print(count)
    return inp, m_out, q_out, r_out


def create_ratio_dict(ratios):
    count = 0
    r = sorted(ratios.keys())
    print(r)
    for i in range(len(r)):
        r_dict[r[i]] = count
        count += 1


if __name__ == '__main__':
    embeddings_filename = "/mnt/povobackup/clic/sandro.pezzelle/corpus-and-vectors/GoogleNews-vectors-negative300.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_dataset_path", type=str, default="lang_dataset/")
    parser.add_argument("--vision_dataset_path", type=str, default="vision_dataset/")
    parser.add_argument("--embeddings_filename", type=str, default=embeddings_filename)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Load language data

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

    # Load visual data

    tr_inds, v_inds, t_inds = read_indices(args.vision_dataset_path)
    ratios = utils.read_qprobs(args.vision_dataset_path)
    tr_size = 11900
    v_size = 1700
    create_ratio_dict(ratios)
    tr_inp, tr_m_out, tr_q_out, tr_r_out = read_images(tr_inds, tr_size)
    v_inp, v_m_out, v_q_out, v_r_out = read_images(v_inds, v_size)
