from __future__ import print_function
import os
import sys
import argparse
import math

# python transform.py train_remap.txt test_remap.txt train_remap_tfidf.txt test_remap_tfidf.txt
# python transform.py train_remap1.txt test_remap1.txt train_tfidf.txt test_tfidf.txt


# assume that label index and feature index start from 0
def tfidf_transform(filename, output_file):
    #construct vocabulary
    vocab = {}
    f = open(filename, 'r')
    g = open(output_file, 'w')

    line_number = 1
    num_docs = 0
    for line in f:
        line = line.strip('\n')
        tmp = line.split(' ')
        if ':' in tmp[0]:
            sys.exit("input format wrong at line " + str(line_number))
        for i in range(1, len(tmp)):
            if tmp[i] == '\n':
                tmp.pop(i)
                continue
            s = tmp[i]
            ss = s.split(':')
            if ss[0] in vocab:
                vocab[ss[0]] += 1
            else:
                vocab[ss[0]] = 1
        line_number += 1
        num_docs += 1

    #tf-idf transformation and idx changing
    line_number = 1
    f.seek(0)
    for line in f:
        line = line.strip('\n')
        tmp = line.split(' ')
        if ':' in tmp[0]:
            sys.exit("input format wrong at line " + str(line_number))
        weighted_square_list = []
        weight_list = []
        for i in range(1, len(tmp)):
            if tmp[i] == '\n':
                tmp.pop(i)
                continue
            s = tmp[i]
            ss = s.split(':')
            ocurInstance = int( round( float(ss[1]) ) )
            totalOccur = vocab[ss[0]]
            weight = 0.0
            if ocurInstance > 0:
                tf = 1 + math.log(ocurInstance)
                idf = math.log(num_docs*1.0/totalOccur)
                weight = tf * idf
            weighted_square_list.append( weight**2 )
            weight_list.append(weight)
        weighted_sqrtroot = math.sqrt( sum(weighted_square_list) )

        #label_info = tmp[0].split(',')
        #new_labels = [ str( int(label) + 1 )  for label in label_info ]
        #tmp[0] = ','.join(new_labels)

        assert (len(weight_list) == (len(tmp) - 1)), "length of weight list doesn't match with number of features"
        for i in range(1, len(tmp)):
            #test data may have missing features
            if tmp[i] == '\n':
                tmp.pop(i)
                continue
            s = tmp[i]
            ss = s.split(':')
            #new_ss = []
            #if remap_flag == 1:
            #    if ss[0] in feature_dict:
            #        ss[0] = feature_dict[ ss[0] ]
            #elif feature_start_flag == 1:
            #    ss[0] = str( (int(ss[0])) + 1 )
            #else:
            #    pass
            if weighted_sqrtroot > 0:
                ss[1] = "{0:.4E}".format( weight_list[i-1] / weighted_sqrtroot )
            tmp[i] = ':'.join(ss)
        newline = ' '.join(tmp)
        if ':' in newline:
            print(newline, file = g)
        else:
            print("werid at line: ", line_number)
        line_number += 1

    f.close()
    g.close()

# def remap(filename):
#     #remapping
#     f = open(filename, 'r')
#     line_number = 1
#     feature_dict = {}
#     label_dict = {}
#     if remap_flag == 1:
#         #f.seek(0)
#         feature_counter = 1
#         label_counter = 1
#         for line in f:
#             tmp = line.split(' ')
#             if ':' in tmp[0]:
#                 sys.exit("input format wrong at line " + str(line_number))
#             label_info = tmp[0].split(',')
#             for label in label_info:
#                 if label not in label_dict:
#                     label_dict[label] = str(label_counter)
#                     label_counter += 1
#             for i in range(1, len(tmp)):
#                 if tmp[i] == '\n':
#                     tmp.pop(i)
#                     continue
#                 s = tmp[i]
#                 ss = s.split(':')
#                 if ss[0] not in feature_dict:
#                     feature_dict[ss[0]] = str(feature_counter)
#                     feature_counter += 1
#
#             print("total number of labels: ", label_counter - 1)
#             print("total number of features: ", feature_counter - 1)
#     f.close()
#     return label_dict, feature_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', help="Train Data")
    parser.add_argument('test_data', help="Test Data")
    parser.add_argument('output_train', help="Output for train")
    parser.add_argument('output_test', help="Output for test")
    #parser.add_argument('-o', '--output_file', action='store', dest='output',
    #                  default="", help="Prefix for the output files")
    #parser.add_argument('-l', '--label_start_index', action='store', dest='l', type=int,
    #                    default=0, help="label start index")
    #parser.add_argument('-f', '--feature_start_index', action='store', dest='f', type=int,
    #                    default=0, help="feature start index")
    #parser.add_argument('-r', '--remap_flag', action='store', dest='r', type=int,
    #                    default=0, help="flag for remapping")

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    output_train = args.output_train
    output_test = args.output_test
    #label_start_flag = args.l
    #feature_start_flag = args.f
    #remapping_flag = args.r

    # if remapping_flag == 1:
    #     label_dict, feature_dict = remap(train_data)
    # else:
    #     label = {}
    #     feature_dict = {}

    tfidf_transform(train_data, output_train)
    print("training data transform complete!")
    tfidf_transform(test_data, output_test)
    print("testing data transform complete!")








    # f = open(filename, 'r')
    # g = open("train_tfidf.txt", 'w')
    # line_number = 1
    # for line in f:
    #     tmp = line.split(' ')
    #     if ':' in tmp[0]:
    #         sys.exit("wrong input")
    #     for i in range(1, len(tmp)):
    #         if tmp[i] == '\n':
    #             tmp.pop(i)
    #             continue
    #         s = tmp[i]
    #         ss = s.split(':')
    #         ss[0] = str( (int(ss[0]) - 1 ) )
    #         tmp[i] = ':'.join(ss)
    #     newline = ' '.join(tmp)
    #     if ':' in newline:
    #         print(newline, file = g)
    #     else:
    #         print("werid at line:%d", line_number)
    #     line_number += 1
