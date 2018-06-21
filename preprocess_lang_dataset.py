import argparse
import unicodecsv as csv
import os
import pickle
from random import shuffle

import numpy as np
from keras.preprocessing.sequence import pad_sequences

import utils

data_path = ''
ratios = {}
r_dict = {}


def load_people(target_index, non_target_index):
    target = data_path + '1900_133.txt'
    non_target = data_path + '1700_133.txt'

    target_names = []
    target_bios = []
    non_target_names = []
    non_target_bios = []

    with open(target) as targx:
        reader = csv.reader(targx, delimiter="\t", encoding="utf-8")
        for row in reader:
            target_names.append(row[2].strip())
            target_bios.append(row[3].strip())

    with open(non_target) as ntargx:
        reader = csv.reader(ntargx, delimiter="\t", encoding="utf-8")
        for row in reader:
            non_target_names.append(row[2].strip())
            non_target_bios.append(row[3].strip())

    people = []
    people_names = []
    people_years = []

    for idx in target_index:
        if idx is not '':
            i = int(idx)
            myperson = target_bios[i].split()[:50]
            people.append(myperson)
            people_names.append(target_names[i])
            people_years.append("1900")

    for idx in non_target_index:
        if idx is not '':
            j = int(idx)
            myperson = non_target_bios[j].split()[:50]
            people.append(myperson)
            people_names.append(non_target_names[j])
            people_years.append("1700")

    pad = 25 - int(len(people))

    for _ in range(pad):
        people.append([])
        people_names.append("#pad#")
        people_years.append("#pad#")

    triples = list(zip(people, people_names, people_years))

    shuffle(triples)

    people, people_names, people_years = zip(*triples)

    people = list(people)
    people_names = list(people_names)
    people_years = list(people_years)

    return people, people_names, people_years


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
        inp_names = [[] for _ in range(size)]
        inp_years = [[] for _ in range(size)]
        m_out = np.zeros((size, 3))
        q_out = np.zeros((size, 9))
        r_out = np.zeros((size, 17))
        count = 0

    with open(split, 'r') as splitfile:
        for n, line in enumerate(splitfile):
            print("Processing line {}/{}".format(n + 1, size))
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

            inpl, people_names, people_years = load_people(target_arr, non_target_arr)
            inp[count].extend(inpl)
            inp_names[count].extend(people_names)
            inp_years[count].extend(people_years)
            count += 1

        return inp, m_out, q_out, r_out, inp_names, inp_years


def create_ratio_dict(ratios):
    count = 0
    r = sorted(ratios.keys())
    for i in range(len(r)):
        r_dict[r[i]] = count
        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    repository_path = args.repository_path
    data_path = args.data_path

    tr, v, tst = read_data(data_path)
    ratios = utils.read_qprobs(repository_path)
    create_ratio_dict(ratios)

    print("Processing training set")
    tr_inp, tr_m_out, tr_q_out, tr_r_out, tr_inp_names, tr_inp_years = load_data(tr)

    print("Processing validation set")
    v_inp, v_m_out, v_q_out, v_r_out, v_inp_names, v_inp_years = load_data(v)

    print("Processing test set")
    t_inp, t_m_out, t_q_out, t_r_out, t_inp_names, t_inp_years = load_data(tst)

    token2id = {}
    id2token = {}

    m_out2id = {
        "less": tuple(np.array([1, 0, 0])),
        "same": tuple(np.array([0, 1, 0])),
        "more": tuple(np.array([0, 0, 1]))
    }
    id2m_out = {y: x for x, y in m_out2id.items()}

    r_out2id = {}
    for r in r_dict:
        r_out = np.zeros((17,))
        r_out[r_dict[r]] = 1
        r_out = tuple(r_out)
        r_out2id[r] = r_out
    id2r_out = {y: x for x, y in r_out2id.items()}

    dataset_tr, dataset_v, dataset_t = [], [], []

    for n, dtp in enumerate(tr_inp):
        for i in dtp:
            authors = []
            for nn, el in enumerate(i):
                author = []
                for token in el:
                    if token not in token2id:
                        i = len(token2id) + 1
                        token2id[token] = i
                        id2token[i] = token
                    author.append(token2id[token])
                authors.append(author)
            authors = pad_sequences(authors, padding='post')
            dataset_tr.append(authors)
    dataset_tr = np.array(dataset_tr)

    for n, dtp in enumerate(v_inp):
        for i in dtp:
            authors = []
            for nn, el in enumerate(i):
                author = []
                for token in el:
                    if token not in token2id:
                        i = len(token2id) + 1
                        token2id[token] = i
                        id2token[i] = token
                    author.append(token2id[token])
                authors.append(author)
            authors = pad_sequences(authors, padding='post')
            dataset_v.append(authors)
    dataset_v = np.array(dataset_v)

    for n, dtp in enumerate(t_inp):
        for i in dtp:
            authors = []
            for nn, el in enumerate(i):
                author = []
                for token in el:
                    if token not in token2id:
                        i = len(token2id) + 1
                        token2id[token] = i
                        id2token[i] = token
                    author.append(token2id[token])
                authors.append(author)
            authors = pad_sequences(authors, padding='post')
            dataset_t.append(authors)
    dataset_t = np.array(dataset_t)

    with open(os.path.join(args.output_path, "index.pkl"), mode="wb") as out_file:
        pickle.dump({
            "token2id": token2id,
            "id2token": id2token,
            "m_out2id": m_out2id,
            "id2m_out": id2m_out,
            "r_out2id": r_out2id,
            "id2r_out": id2r_out
        }, out_file)

    with open(os.path.join(args.output_path, "train.pkl"), mode="wb") as out_file:
        pickle.dump({
            "dataset_tr": dataset_tr,
            "tr_m_out": tr_m_out,
            "tr_q_out": tr_q_out,
            "tr_r_out": tr_r_out,
            "dataset_tr_names": tr_inp_names,
            "dataset_tr_years": tr_inp_years
        }, out_file)

    with open(os.path.join(args.output_path, "test.pkl"), mode="wb") as out_file:
        pickle.dump({
            "dataset_t": dataset_t,
            "t_m_out": t_m_out,
            "t_q_out": t_q_out,
            "t_r_out": t_r_out,
            "dataset_v_names": v_inp_names,
            "dataset_v_years": v_inp_years
        }, out_file)

    with open(os.path.join(args.output_path, "valid.pkl"), mode="wb") as out_file:
        pickle.dump({
            "dataset_v": dataset_v,
            "v_m_out": v_m_out,
            "v_q_out": v_q_out,
            "v_r_out": v_r_out,
            "dataset_t_names": t_inp_names,
            "dataset_t_years": t_inp_years
        }, out_file)
