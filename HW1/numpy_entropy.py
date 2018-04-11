#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    d = {}
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            if line in d:
                d[line] += 1
            else:
                d[line] = 1

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    m = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            line = line.split()
            if line[0] in m:
                m[line[0]] += float(line[1])
            else:
                m[line[0]] = float(line[1])


    arr_d = []
    for word in d:
        if word not in m:
            m[word] = 0.0
    arr_m = []

    to_delete = []
    for word in m:
        if word not in d:
            to_delete.append(word)

    for word in to_delete:
        m.pop(word)

    for word in sorted(d):
        arr_d.append(float(d[word]) / sum(d.values()))
    arr_d = np.array(arr_d)
    for word in sorted(m):
        arr_m.append(float(m[word]) / sum(m.values()))
    arr_m = np.array(arr_m)

    entropy = np.sum(-arr_d * np.log(arr_d))
    cross_entropy = np.sum(-arr_d * np.log(arr_m))
    dkl = cross_entropy - entropy

    print("{:.2f}".format(entropy))
    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(dkl))


