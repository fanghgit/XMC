from __future__ import print_function
import os
import sys
import argparse
import math


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data', help="input test_data")
    parser.add_argument('label_file', help="output label_file")
    parser.add_argument('test_fakelabel', help="output test file with fake label")

    args = parser.parse_args()
    test_data = args.test_data
    label_file = args.label_file
    test_fakelabel = args.test_fakelabel

    f = open(test_data, 'r')
    g_label = open(label_file, 'w')
    g_test = open(test_fakelabel, 'w')

    for line in f:
        line = line.strip('\n')
        tmp = line.split(' ')
        # write to label file
        print(tmp[0], file = g_label)
        tmp[0] = "0"
        newline = ' '.join(tmp)
        #write to test file with fake label
        print(newline, file = g_test)

    f.close()
    g_label.close()
    g_test.close()
