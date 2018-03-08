from __future__ import print_function
import os
import sys
import argparse
import math


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('actual_data_file', help="Test Data")
    parser.add_argument('predicted_data_file', help="Predicted Result")

    args = parser.parse_args()
    actual_data_file = args.actual_data_file
    predicted_data_file = args.predicted_data_file

    k1 = 1
    k2 = 3
    k3 = 5
    p1 = 0
    p3 = 0
    p5 = 0
    ndcg1 = 0
    ndcg3 = 0
    ndcg5 = 0

    f = open(actual_data_file, 'r')
    g = open(predicted_data_file, 'r')

    n_test = 0
    while(True):
        line_actual = f.readline()
        line_predict = g.readline()
        if len(line_actual) == 0 and len(line_predict) == 0:
            break
        elif len(line_actual)*len(line_predict) == 0:
            sys.exit("number of predtion and testing data does not match!")
        else:
            pass

        if line_actual == " \n":
            print("test data with no label")
            continue

        n_test += 1
        line_actual = line_actual.strip('\n')
        line_predict = line_predict.strip('\n')

        tmp = line_predict.split(' ')
        pred_list = [int(x.split(':')[0]) for x in tmp if len(x) != 0 ]
        tmp_actual = line_actual.split(',')
        actual_list = [int(x) for x in tmp_actual]
        print(n_test)

        deno1 = 0
        deno3 = 0
        deno5 = 0
        dcg1 = 0
        dcg3 = 0
        dcg5 = 0
        for i in range(k1):
            if pred_list[i] in actual_list:
                p1 += 1
                p3 += 1
                p5 += 1
                dcg1 += math.log(2)/math.log(i+2)
                dcg3 += math.log(2)/math.log(i+2)
                dcg5 += math.log(2)/math.log(i+2)
        for i in range(k1+1, k2):
            if pred_list[i] in actual_list:
                p3 += 1
                p5 += 1
                dcg3 += math.log(2)/math.log(i+2)
                dcg5 += math.log(2)/math.log(i+2)
        for i in range(k2+1,k3):
            if pred_list[i] in actual_list:
                p5 += 1
                dcg5 += math.log(2)/math.log(i+2)

        deno1 = sum( [math.log(2)/math.log(i+2) for i in range(min(k1, len(actual_list)) ) ] )
        deno3 = sum( [math.log(2)/math.log(i+2) for i in range(min(k2, len(actual_list)) ) ] )
        deno5 = sum( [math.log(2)/math.log(i+2) for i in range(min(k3, len(actual_list)) ) ] )
        ndcg1 += dcg1 / deno1
        ndcg3 += dcg3 / deno3
        ndcg5 += dcg5 / deno5

    p1 = (p1*100.0)/(1.0*n_test*k1)
    p3 = (p3*100.0)/(1.0*n_test*k2)
    p5 = (p5*100.0)/(1.0*n_test*k3)
    ndcg1 = (ndcg1*100.0)/(1.0*n_test)
    ndcg3 = (ndcg3*100.0)/(1.0*n_test)
    ndcg5 = (ndcg5*100.0)/(1.0*n_test)

    f.close()
    g.close()

    print("precision at 1: ", p1)
    print("precision at 3: ", p3)
    print("precision at 5: ", p5)
    print("ndcg at 1: ", ndcg1)
    print("ndcg at 3: ", ndcg3)
    print("ndcg at 5: ", ndcg5)
