from __future__ import print_function
import os
import sys
import argparse
import math

# python remap.py train.txt test.txt train_remap.txt test_remap.txt -r 1
# python remap.py train.txt test.txt train_remap1.txt test_remap1.txt -l 1 -f 1


def get_map(filename):
    #remapping
    f = open(filename, 'r')
    line_number = 1
    feature_dict = {}
    label_dict = {}
    if remap_flag == 1:
        #f.seek(0)
        feature_counter = 1
        label_counter = 1
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            tmp = line.split(' ')
            if ':' in tmp[0]:
                sys.exit("input format wrong at line " + str(line_number))
            label_info = tmp[0].split(',')
            if len(tmp[0]) != 0:
                for label in label_info:
                    if label not in label_dict:
                        label_dict[label] = str(label_counter)
                        label_counter += 1
            for i in range(1, len(tmp)):
                if tmp[i] == '\n':
                    tmp.pop(i)
                    continue
                s = tmp[i]
                ss = s.split(':')
                if ss[0] not in feature_dict:
                    feature_dict[ss[0]] = str(feature_counter)
                    feature_counter += 1

        print("total number of labels: ", label_counter - 1)
        print("total number of features: ", feature_counter - 1)
    f.close()
    return label_dict, feature_dict


def remap_train(input_file, output, label_dict, feature_dict, label_start_flag, feature_start_flag, remap_flag):
    f = open(input_file, 'r')
    g = open(output, 'w')

    line_number = 1
    for line in f:
        line = line.strip('\n')
        line = line.strip('r')
        tmp = line.split(' ')
        if ':' in tmp[0]:
            sys.exit("input format wrong at line " + str(line_number))
        #tmp_new = []
        # label remap
        if remap_flag == 1 and len(tmp[0]) != 0:
            label_info = tmp[0].split(',')
            label_info_new = [ label_dict[label] for label in label_info ]
            tmp[0] = ','.join(label_info_new)
        elif label_start_flag == 1 and len(tmp[0]) != 0:
            label_info = tmp[0].split(',')
            label_info_new = [ str( int(label) + 1 ) for label in label_info ]
            tmp[0] = ','.join(label_info_new)
        else:
            pass
        # feature remap
        tmp_dict = {}
        for i in range(1, len(tmp)):
            if tmp[i] == '\n':
                tmp.pop(i)
                continue
            s = tmp[i]
            ss = s.split(':')
            if remap_flag == 1:
                #ss[0] = feature_dict[ ss[0] ]
                tmp_dict[ int(feature_dict[ ss[0] ]) ] = ss[1]
            elif feature_start_flag == 1:
                tmp_dict[ int(ss[0]) + 1 ] = ss[1]
            else:
                pass
            # tmp[i] = ':'.join(ss)
        newline_list = []
        newline_list.append( tmp[0] )
        keys = sorted(tmp_dict.keys())
        for key in keys:
            newline_list.append( str(key)+":"+tmp_dict[key] )
        newline = ' '.join(newline_list)
        if ':' in newline:
            print(newline, file = g)
        else:
            print("no feature at line: ", line_number)
        line_number += 1

    f.close()
    g.close()

def remap_test(input_file, output, label_dict, feature_dict, label_start_flag, feature_start_flag, remap_flag):
    f = open(input_file, 'r')
    g = open(output, 'w')

    line_number = 1
    cur_label = len(label_dict)
    for line in f:
        line = line.strip('\n')
        line = line.strip('\r')
        tmp = line.split(' ')
        if ':' in tmp[0]:
            sys.exit("input format wrong at line " + str(line_number))
        #tmp_new = []
        # label remap
        if remap_flag == 1 and len(tmp[0]) != 0:
            label_info = tmp[0].split(',')
            label_info_new = []
            for label in label_info:
                if label in label_dict:
                    label_info_new.append( label_dict[label] )
                else:  # label that not appear in training set
                    cur_label += 1
                    label_dict[label] = str(cur_label)
                    label_info_new.append( str(cur_label) )
                    print("new label in test set: " + label)
            tmp[0] = ','.join(label_info_new)
        elif label_start_flag == 1 and len(tmp[0]) != 0:
            label_info = tmp[0].split(',')
            label_info_new = [ str( int(label) + 1 ) for label in label_info ]
            tmp[0] = ','.join(label_info_new)
        else:
            pass

        # feature remap
        tmp_dict = {}
        for i in range(1, len(tmp)):
            if tmp[i] == '\n':
                tmp.pop(i)
                continue
            s = tmp[i]
            ss = s.split(':')
            if remap_flag == 1:
                if ss[0] in feature_dict:
                    tmp_dict[ int( feature_dict[ss[0] ] ) ] = ss[1]
                else: # debug
                    if line_number == 1:
                        print("missing feature in training data: " + ss[0])
                #else: # features that not appears in training set
            elif feature_start_flag == 1:
                tmp_dict[ int(ss[0]) + 1 ] = ss[1]
        newline_list = []
        newline_list.append( tmp[0] )
        keys = sorted(tmp_dict.keys())
        for key in keys:
            newline_list.append( str(key)+":"+tmp_dict[key] )
        newline = ' '.join(newline_list)
        # debug
        #if line_number == 1:
            #print("num features in original data: ", line.count(':'))
            #print("num features in new data: ", newline.count(':'))
            #print(newline_list)
            #print(newline)
        if ':' in newline:
            print(newline, file = g)
        else:
            print("no feature at line: ", line_number)
        line_number += 1

    print("total num of labels in train and test: ", len(label_dict))
    f.close()
    g.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', help="Train Data")
    parser.add_argument('test_data', help="Test Data")
    parser.add_argument('output_train', help="Output for train")
    parser.add_argument('output_test', help="Output for test")
    parser.add_argument('-l', '--label_start_index', action='store', dest='l', type=int,
                       default=0, help="label start index")
    parser.add_argument('-f', '--feature_start_index', action='store', dest='f', type=int,
                       default=0, help="feature start index")
    parser.add_argument('-r', '--remap_flag', action='store', dest='r', type=int,
                       default=0, help="flag for remapping")

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    output_train = args.output_train
    output_test = args.output_test
    label_start_flag = args.l
    feature_start_flag = args.f
    remap_flag = args.r

    label_dict, feature_dict = get_map(train_data)

    remap_train(train_data, output_train, label_dict, feature_dict, label_start_flag, feature_start_flag, remap_flag)
    print("training remap finished!")
    remap_test(test_data, output_test, label_dict, feature_dict, label_start_flag, feature_start_flag, remap_flag)
    print("testing remap finished!")
