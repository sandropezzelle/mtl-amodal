def read_qprobs(path):
    count = 0
    ratios = {}
    ratio_l = []

    with open(path + 'Q-probabilities.txt', 'r') as in_file:
        for line in in_file:
            els = line.strip().split('\t')

            if count == 0:
                for el in els[1:]:
                    ratios[el] = {}

                ratio_l = els[1:]

                for el in ratios:
                    for i in range(9):
                        ratios[el][str(i)] = 0.0
            else:
                ind = els[0]

                for i in range(17):
                    val = els[2 + i]
                    r = ratio_l[i]
                    ratios[r][ind] = val

            count += 1

    return ratios
