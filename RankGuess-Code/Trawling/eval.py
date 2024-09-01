import matplotlib.pyplot as plt
import pandas as pd

from sample import argparser

def eval(generate, test):
    generate_file = open(generate, 'r')
    test_file = open(test, 'r')
    test_dict = dict()
    test_cnt = 0
    while True:
        line = test_file.readline()
        if not line:
            break
        test_cnt += 1
        line = line[:-1]
        if line in test_dict:
            test_dict[line] += 1
        else:
            test_dict[line] = 1
    cnt = 0
    s = 0
    x = list()
    y = list()
    while True:
        line = generate_file.readline()
        if not line:
            break
        line = line[:-1]
        if line in test_dict:
            s += test_dict[line]
        cnt += 1
        if cnt % 1000 == 0:
            # print("INFO: evaluate {} password, crack {}, {}".format(cnt, s, s / test_cnt))
            x.append(cnt)
            y.append(s/test_cnt)                                                                                                                                         
    # print("INFO: evaluate {} password, crack {}, {}".format(cnt, s, s / test_cnt))
    #         print("{} {} {}".format(cnt, s, s / test_cnt))
    Data = pd.DataFrame({'x':x, 'y':y})
    Data.to_csv('', index=False)

if __name__ == '__main__':
    eval('', "")
