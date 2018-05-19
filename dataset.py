from random import shuffle

import numpy as np

data_path = ''
ratios = {}
r_dict = {}


def load_people(t_idx, nont_idx):
    targ = data_path + '1900_133.txt'
    ntarg = data_path + '1700_133.txt'

    with open(targ, 'r') as targx:
        reader = [line.split('\t')[3] for line in targx]

    with open(ntarg, 'r') as ntargx:
        nreader = [line.split('\t')[3] for line in ntargx]

    people = []

    for idx in t_idx:
        if idx is not '':
            i = int(idx)
            myperson = reader[i].split()[0:50]
            people.append(myperson)

    for idx in nont_idx:
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
