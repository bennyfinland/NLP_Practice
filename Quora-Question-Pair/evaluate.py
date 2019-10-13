import sys
import os
import argparse
import scipy as sp

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--actual", help="Actual result file")
    parser.add_argument("-p", "--predict", help="Predict result file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    predict_list = list()
    for line_cnt, line in enumerate(open(args.predict, 'r').xreadlines()):
        if line_cnt == 0:
            continue
        predict_list.append(float(line.strip().rsplit(' ', 1)[1]))

    actual_list = list()
    for line in open(args.actual, 'r').xreadlines():
        actual_list.append(float(line.strip().split(' ', 1)[0]))

    #pred = [1,0,1,0]
    #act = [1,0,1,0]
    #print logloss(act, pred)
    print logloss(actual_list, predict_list)
