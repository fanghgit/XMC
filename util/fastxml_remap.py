from __future__ import print_function
import os
import sys
import argparse
import math

# python fastxml_remap.py ../../../data/eurlex/train-remapped-tfidf.txt ../../../data/eurlex/test-remapped-tfidf.txt Data/EUR-Lex/train_tf.txt Data/EUR-Lex/test_tf.txt Data/EUR-Lex/train_lbl.txt Data/EUR-Lex/test_lbl.txt

def remap(input, output_ft, output_lbl):
    f = open(input, 'r')
    g_ft = open(output_ft, 'w')
    g_lbl = open(output_lbl, 'w')

    line_number = 1
    for line in f:
        line = line.strip('\n')
        line = line.strip('\r')
        tmp = line.split(' ')
        if ':' in tmp[0]:
            sys.exit("input format wrong at line " + str(line_number))
        # process labels
        if len(tmp[0]) != 0:
            labels = tmp[0].split(',')
            new_labels = [ (str( int(x) - 1 )+":1") for x in labels ]
            label_line = ' '.join(new_labels)
        else:
            label_line = ""

        print(label_line, file = g_lbl)

        #newline = ""
        for i in range(1, len(tmp)):
            if tmp[i] == '\n':
                tmp.pop(i)
                continue
            if len(tmp[i]) == 0:
                continue
            s = tmp[i]
            ss = s.split(':')
            ss[0] = str(int(ss[0])-1)
            tmp[i] = ':'.join(ss)
        tmp.pop(0)
        newline = ' '.join(tmp)
        print(newline, file = g_ft)


    f.close()
    g_ft.close()
    g_lbl.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', help="Train Data")
    parser.add_argument('test_data', help="Test Data")
    parser.add_argument('output_train_ft', help="Output for train feature")
    parser.add_argument('output_test_ft', help="Output for test feature")
    parser.add_argument('output_train_lbl', help="Output for train label")
    parser.add_argument('output_test_lbl', help="Output for test label")

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    output_train_ft = args.output_train_ft
    output_train_lbl = args.output_train_lbl
    output_test_ft = args.output_test_ft
    output_test_lbl = args.output_test_lbl

    remap(train_data, output_train_ft, output_train_lbl)
    print("training data remap complete!")
    remap(test_data, output_test_ft, output_test_lbl)
    print("testing data remap complete!")
