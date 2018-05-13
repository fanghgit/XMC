DiSMEC - Distributed Sparse Machines for Extreme multi-label classification problems

=====================
Some features of DiSMEC
=====================
- C++ code built over the Liblinear-64 [1] bit code-base to handle multi-label datasets in LibSVM format, and handle parallel training and prediction using openMP

- Handles extreme multi-label classification problems consisting of millions of labels in datasets which can be downloaded from Extreme Classification Repository

- Takes only a few minutes on EURLex-4K (eurlex) dataset consisting of about 4,000 labels and a few hours on WikiLSHTC-325K datasets consisting of about 325,000 labels

- Learns models in the batch of 1000 (by default, can be changed to suit your settings) labels

- Allows two-folds parallel training (a) A single batch of labels (say 1000 labels) is learnt in parallel using openMP while exploiting multiple cores on a single machine/node/computer (b) Can be invoked on multiple computers/machines/nodes simultaneously such that the successive launch starts from the next batch of 1000 labels(see -i option below)

- Tested on 64-bit Ubuntu only, not not very clean code but runs

Detailed instructions for using DiSMEC are given below.  These instructions are to reproduce the results for Eurlex data only.
For more details, check our paper "DiSMEC - Distributed Sparse Machines for Extreme Multi-label Classification" [2] , or email me if anything is unclear

===================
CONTENTS
===================
There are directories
1) ./dismec contains the DiSMEC code

2) ./eurlex consists of data for EURLex-4k downloaded from XMC repository

3) ./prepostprocessing consists of Java code for (a) pre-processing data to get into tf-idf format and remapping labels and features, and (b) Evaluation of precision@k and nDCG@k corresponding to the prediction results.

===================
USAGE
===================

Data Pre-processing (in Java)
0) Download the eurlex dataset from XMC repository, and remove the first line from the train and test files downloaded, call them train.txt and test.txt

1) Change feature ID's so that they start from 1..to..number_of_features, using the code provided in FeatureRemapper.java using the following command
javac FeatureRemapper.java
java FeatureRemapper ../eurlex/train.txt ../eurlex/train-remapped.txt ../eurlex/test.txt ../eurlex/test-remapped.txt

2) Convert to tf-idf format using the code in file TfIdfCalculator.java
javac TfIdfCalculator.java
java TfIdfCalculator ../eurlex/train-remapped.txt ../eurlex/train-remapped-tfidf.txt ../eurlex/test-remapped.txt ../eurlex/test-remapped-tfidf.txt


java TfIdfCalculator ../../data/RCV1/rcv1_train_dis_remapped.txt ../../data/RCV1/rcv1_train_dis_remapped_tfidf.txt ../../data/RCV1/rcv1_test_dis_remapped.txt ../../data/RCV1/rcv1_test_dis_remapped_tfidf.txt


3) Change labels ID's so that they also start from 1..to..number_of_labels, using the code provided in LabelRelabeler.java
javac LabelRelabeler.java
java LabelRelabeler ../eurlex/train-remapped-tfidf.txt ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/test-remapped-tfidf.txt ../eurlex/test-remapped-tfidf-relabeled.txt ../eurlex/label_map.txt

===================

Building DiSMEC
Just run make command in the ../dismec/ directory. This will build the train and predict executable

===================

Training model with DiSMEC (in C++)

Training models with DiSMEC with 2-level parallelism
// make the directory to write the model files
mkdir ../eurlex/models
../dismec/train -s 2 -B 1 -i 1 ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/models/1.model
../dismec/train -s 2 -B 1 -i 2 ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/models/2.model
../dismec/train -s 2 -B 1 -i 3 ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/models/3.model
../dismec/train -s 2 -B 1 -i 4 ../eurlex/train-remapped-tfidf-relabeled.txt ../eurlex/models/4.model

If run in parallel on multiple machines, the models might take around 5 mins to build on this dataset.

===================

Predicting with DiSMEC in parallel (in C++)

Since the base Liblinear code does not understand the comma separated labels. We need to zero out labels in the test file, and put that in a separate file (called GS.txt) consisting of only the labels.
javac LabelExtractor.java
java LabelExtractor ../eurlex/test-remapped-tfidf-relabeled.txt ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/GS.txt


mkdir ../eurlex/output // make the directory to write output files
../dismec/predict ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/models/1.model ../eurlex/output/1.out
../dismec/predict ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/models/2.model ../eurlex/output/2.out
../dismec/predict ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/models/3.model ../eurlex/output/3.out
../dismec/predict ../eurlex/test-remapped-tfidf-relabeled-zeroed.txt ../eurlex/models/4.model ../eurlex/output/4.out

===================

Performance evaluation (in Java)

Computation of Precision@k and nDCG@k for k=1,3,5
Now, we need to get final top-1, top-3 and top-5 from the output of individual models. This is done by the following :

mkdir ../eurlex/final-output
javac DistributedPredictor.java
java DistributedPredictor ../eurlex/output/ ../eurlex/final-output/top1.out ../eurlex/final-output/top3.out ../eurlex/final-output/top5.out

javac MultiLabelMetrics.java
java MultiLabelMetrics ../eurlex/GS.txt ../eurlex/final-output/top1.out ../eurlex/final-output/top3.out ../eurlex/final-output/top5.out

The expected output should be something like this:
 precision at 1 is 82.51380489087562
 precision at 3 is 69.48023490227014
 precision at 5 is 57.94372863528793

 ndcg at 1 is 82.51380489087562
 ndcg at 3 is 72.89528751985883
 ndcg at 5 is 67.05303357275555

===================

Other Datasets from XMC repository:
If you would like to build for another dataset, please change the number of labels and replace with appropriate number at line number 2301, and then run make

long long nr_class = 3786; //3786 for eurlex

If you would like to change the batch size (1000 by default) for you settings, please replace with appropriate number at line number 2427, and then run make

int batchSize = 1000;

======================
References
[1] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. LIBLINEAR: A library for large linear classification Journal of Machine Learning Research 9(2008), 1871-1874.
[2] R. Babbar, B. Schölkopf. DiSMEC: Distributed Sparse Machines for Extreme Multi-label Classification, WSDM 2017



line-search, zeros initialization:   575.089966 sec
line-search, all neg initialization: 248.850006 sec
trust region, zeros initialization:  655.139954 sec



-B 1 -e 0.001

recision at 1:  82.8030502235
precision at 3:  69.5152949426
precision at 5:  57.9700236655
ndcg at 1:  82.8030502235
ndcg at 3:  72.9064572919
ndcg at 5:  67.0386701337

dismec training time: 421.980011


-m 1 -B 1 -e 0.001

precision at 1: 82.48884
 precision at 3: 69.21327
 precision at 5: 57.52166
ndcg at 1: 82.48884
 ndcg at 3: 72.63272
 ndcg at 5: 66.64887

total training time: 216.770004
naive: 436.110016



-e 0.0001
line-search, MST initialization: 181.520004 sec,  eu2.model
line-search, all neg initialization: 187.679993 sec, eu3.model

-e 0.001
line-search, MST initialization: 144.610001 sec,
line-search, all neg initialization: 142.830002 sec

###
Reading data
training model, using w
save model, w+1
loading model, w+1
prediction, add 1
###



wiki10: -e 0.001

load model complete, time spent: 362.180000 sec
test read complete!
load test file complete, time spent: 1.110000 sec
time spent on prediction: 234.920000 sec
precision at 1: 84.55260
 precision at 3: 74.27952
 precision at 5: 65.22370
ndcg at 1: 84.55260
 ndcg at 3: 76.72839
 ndcg at 5: 69.84375









EURLex: parallel

1 thread: 222.503462

10 threads: 30.204160

20 threads: 21.546310


amazon13k: -B 1 -e 1.0 eps = 0.001
time spent on prediction: 428.650000 sec
precision at 1: 92.94092
 precision at 3: 76.56794
 precision at 5: 60.92652
ndcg at 1: 92.94092
 ndcg at 3: 85.67381
 ndcg at 5: 82.84235

amazon13k: -B 1 -eps 0.0001

 precision at 1: 93.76691
  precision at 3: 78.68888
  precision at 5: 62.58346
 ndcg at 1: 93.76691
  ndcg at 3: 87.56113
  ndcg at 5: 84.66369

amazon13k: -B 1 -e 1.0 eps = 0.0001
precision at 1: 93.75354
 precision at 3: 78.89000
 precision at 5: 63.66319
ndcg at 1: 93.75354
 ndcg at 3: 87.70111
 ndcg at 5: 85.50187



wiki325 -B 1 -e 0.0001

precision at 1: 36.61527
 precision at 3: 20.95098
 precision at 5: 14.68981
ndcg at 1: 36.61527
 ndcg at 3: 28.20789
 ndcg at 5: 26.25156


wiki325 -B -1 -e 0.00001

no results






tree statistics:

EURLex:
number of subtrees: 2072
height of the whole tree: 5

-e 1.0 0.001


AmazonCat13k:
number of subtrees: 9510
height of the whole tree: 8

-e 1.0 0.0001

wiki31k
number of subtrees: 9816
height of the whole tree: 10

wiki325k
number of subtrees: 207331
height of the whole tree: 37

Amazon670k
number of subtrees: 235879
height of the whole tree: 23


1. PPDSparse: algo and code
2. SGD
3. try different data

58854
3600 min

training for FastXML
python fastxml_remap.py ./Data/AmazonCat13k/amazonCat_train.txt ./Data/AmazonCat13k/amazonCat_test.txt ./Data/AmazonCat13k/trn_X_Xf.txt ./Data/AmazonCat13k/tst_X_Xf.txt ./Data/AmazonCat13k/trn_X_Y.txt ./Data/AmazonCat13k/tst_X_Y.txt
python fastxml_remap.py ./Data/wiki10/wiki10_train.txt ./Data/wiki10/wiki10_test.txt ./Data/wiki10/trn_X_Xf.txt ./Data/wiki10/tst_X_Xf.txt ./Data/wiki10/trn_X_Y.txt ./Data/wiki10/tst_X_Y.txt
python fastxml_remap.py ./Data/Amazon670k/amazon_train.txt ./Data/Amazon670k/amazon_test.txt ./Data/Amazon670k/trn_X_Xf.txt ./Data/Amazon670k/tst_X_Xf.txt ./Data/Amazon670k/trn_X_Y.txt ./Data/Amazon670k/tst_X_Y.txt
python fastxml_remap.py ./Data/AmazonCat14k/amazonCat-14K_train.txt ./Data/AmazonCat14k/amazonCat-14K_test.txt ./Data/AmazonCat14k/trn_X_Xf.txt ./Data/AmazonCat14k/tst_X_Xf.txt ./Data/AmazonCat14k/trn_X_Y.txt ./Data/AmazonCat14k/tst_X_Y.txt


./fastXML_train ../Sandbox/Data/EUR-Lex/trn_X_Xf.txt ../Sandbox/Data/EUR-Lex/trn_X_Y.txt ../Sandbox/Results/EUR-Lex/model -T 10 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 10

./fastXML_test ../Sandbox/Data/EUR-Lex/tst_X_Xf.txt ../Sandbox/Results/EUR-Lex/score_mat.txt ../Sandbox/Results/EUR-Lex/model

./multiTrain -s 3 ../data/eurlex/train_remap_tfidf.txt models/eu.model

./multiPred3 ../data/AmazonCat13k/test_remap_tfidf.txt models/amazon13k.model

DISMEC:
./train -B 1 -s 2 -P 10 -x 0 -e 1.0 0.001 ../data/eurlex/train_remap_tfidf.txt models/eu.model

./train -B 1 -s 2 -e 0.001 ../../data/eurlex/train_remap_tfidf.txt models/eu.model



function P = helper(score_mat,true_mat,K)
        num_inst = size(score_mat,2);
        num_lbl = size(score_mat,1);

        P = zeros(K,1);
        rank_mat = sort_sparse_mat(score_mat);

        for k=1:K
                mat = rank_mat;
                #mat(rank_mat>k) = 0;

                [i,j,s] = find(mat);
                [m,n] = size(mat);
                idx = i <= m;
                i = i(idx);
                j = j(idx);
                s = s(idx);

                mat = sparse(i,j,s,m,n);

                mat = spones(mat);
                mat = mat.*true_mat;
                num = sum(mat,1);

                P(k) = mean(num/k);
        end
end



addpath(genpath('../Tools'));
trn_X_Y = read_text_mat('../Sandbox/Data/AmazonCat14k/trn_X_Y.txt');
tst_X_Y = read_text_mat('../Sandbox/Data/AmazonCat14k/tst_X_Y.txt');
score_mat = read_text_mat('../Sandbox/Results/AmazonCat14k/score_mat.txt');


m = 10
n = 10
pos = {}
for i in range(1,m+1):
  pos[i] = +1

pos[m+1] = 0

for j in range(m+2, m+n+2):
  pos[j] = -1

class




./linesearch_parallel/train -B 1 -P 1 -x 0 -e 1.0 0.0001 data/eurlex/train_remap_tfidf.txt linesearch_parallel/models/eu.model


L2L1: 132.016738 s

LogL1: 161.594727

LogL2: 113.755924s
LogL2: 48.276696s


GD: 357.427042 s
GDx2: 440.990475
GDx1: 453.254421




kk: 1022
*********** accuracy test begins ************
randomized CD iter: 200
#pos=5, #ran_iter=200, w_nnz=5001, a_nnz=1220, time=2.36883
optimal obj: 90.3868
# iter: 1 time: 0.009714, primal obj: 6121.6
# iter: 2 time: 0.0206184, primal obj: 3311.2
# iter: 3 time: 0.032618, primal obj: 1249.13
# iter: 4 time: 0.0459973, primal obj: 602.68
# iter: 5 time: 0.0624896, primal obj: 255.497
# iter: 6 time: 0.0841072, primal obj: 135.934
# iter: 7 time: 0.10689, primal obj: 104.577
# iter: 8 time: 0.119107, primal obj: 94.7244
# iter: 9 time: 0.131438, primal obj: 91.8145
# iter: 10 time: 0.143601, primal obj: 91.0076
# iter: 11 time: 0.156314, primal obj: 90.669
# iter: 12 time: 0.168683, primal obj: 90.5142
# iter: 13 time: 0.180932, primal obj: 90.4798
# iter: 14 time: 0.193043, primal obj: 90.4403
# iter: 15 time: 0.205292, primal obj: 90.4253
# iter: 16 time: 0.217515, primal obj: 90.4164
# iter: 17 time: 0.229698, primal obj: 90.4135
# iter: 18 time: 0.241866, primal obj: 90.411
# iter: 19 time: 0.253994, primal obj: 90.411
# iter: 20 time: 0.26632, primal obj: 90.4104
# iter: 21 time: 0.27834, primal obj: 90.4105
# iter: 22 time: 0.290422, primal obj: 90.4096
# iter: 23 time: 0.302745, primal obj: 90.4094
# iter: 24 time: 0.314833, primal obj: 90.4093
# iter: 25 time: 0.32694, primal obj: 90.4093
# iter: 26 time: 0.33921, primal obj: 90.4093
# iter: 27 time: 0.351534, primal obj: 90.4093
# iter: 28 time: 0.363938, primal obj: 90.4093
# iter: 29 time: 0.37604, primal obj: 90.4093
# iter: 30 time: 0.387994, primal obj: 90.4093
# iter: 31 time: 0.400192, primal obj: 90.4093
# iter: 32 time: 0.412152, primal obj: 90.4093
# iter: 33 time: 0.424074, primal obj: 90.4093
# iter: 34 time: 0.436016, primal obj: 90.4093
# iter: 35 time: 0.448136, primal obj: 90.4093
# iter: 36 time: 0.460039, primal obj: 90.4093
# iter: 37 time: 0.472058, primal obj: 90.4093
# iter: 38 time: 0.484177, primal obj: 90.4093
# iter: 39 time: 0.496177, primal obj: 90.4093
# iter: 40 time: 0.508163, primal obj: 90.4093
# iter: 41 time: 0.520065, primal obj: 90.4093
# iter: 42 time: 0.531968, primal obj: 90.4093
# iter: 43 time: 0.544073, primal obj: 90.4093
# iter: 44 time: 0.556258, primal obj: 90.4093
# iter: 45 time: 0.568387, primal obj: 90.4093
# iter: 46 time: 0.580394, primal obj: 90.4093
# iter: 47 time: 0.592361, primal obj: 90.4093
# iter: 48 time: 0.604507, primal obj: 90.4093
# iter: 49 time: 0.616595, primal obj: 90.4093
# iter: 50 time: 0.628628, primal obj: 90.4093
# iter: 51 time: 0.640609, primal obj: 90.4093
# iter: 52 time: 0.652622, primal obj: 90.4093
# iter: 53 time: 0.664486, primal obj: 90.4093
# iter: 54 time: 0.676622, primal obj: 90.4093
# iter: 55 time: 0.688498, primal obj: 90.4093
# iter: 56 time: 0.700701, primal obj: 90.4093
# iter: 57 time: 0.712515, primal obj: 90.4093
# iter: 58 time: 0.724535, primal obj: 90.4093
# iter: 59 time: 0.736571, primal obj: 90.4093
# iter: 60 time: 0.748703, primal obj: 90.4093
# iter: 61 time: 0.760711, primal obj: 90.4093
# iter: 62 time: 0.772613, primal obj: 90.4093
# iter: 63 time: 0.784534, primal obj: 90.4093
# iter: 64 time: 0.796295, primal obj: 90.4093
# iter: 65 time: 0.808298, primal obj: 90.4093
# iter: 66 time: 0.82039, primal obj: 90.4093
# iter: 67 time: 0.832396, primal obj: 90.4093
# iter: 68 time: 0.844419, primal obj: 90.4093
# iter: 69 time: 0.856721, primal obj: 90.4093
# iter: 70 time: 0.868842, primal obj: 90.4093
# iter: 71 time: 0.880932, primal obj: 90.4093
# iter: 72 time: 0.89276, primal obj: 90.4093
# iter: 73 time: 0.904756, primal obj: 90.4093
# iter: 74 time: 0.917031, primal obj: 90.4093
# iter: 75 time: 0.928956, primal obj: 90.4093
# iter: 76 time: 0.940829, primal obj: 90.4093
# iter: 77 time: 0.952946, primal obj: 90.4093
# iter: 78 time: 0.964977, primal obj: 90.4093
# iter: 79 time: 0.976902, primal obj: 90.4093
# iter: 80 time: 0.988745, primal obj: 90.4093
# iter: 81 time: 1.00073, primal obj: 90.4093
# iter: 82 time: 1.01264, primal obj: 90.4093
# iter: 83 time: 1.02445, primal obj: 90.4093
# iter: 84 time: 1.0362, primal obj: 90.4093
# iter: 85 time: 1.048, primal obj: 90.4093
# iter: 86 time: 1.05986, primal obj: 90.4093
# iter: 87 time: 1.07169, primal obj: 90.4093
# iter: 88 time: 1.08358, primal obj: 90.4093
# iter: 89 time: 1.09526, primal obj: 90.4093
# iter: 90 time: 1.10739, primal obj: 90.4093
# iter: 91 time: 1.1192, primal obj: 90.4093
# iter: 92 time: 1.13093, primal obj: 90.4093
# iter: 93 time: 1.14261, primal obj: 90.4093
# iter: 94 time: 1.15431, primal obj: 90.4093
# iter: 95 time: 1.16632, primal obj: 90.4093
# iter: 96 time: 1.17789, primal obj: 90.4093
# iter: 97 time: 1.18981, primal obj: 90.4093
# iter: 98 time: 1.20173, primal obj: 90.4093
# iter: 99 time: 1.21366, primal obj: 90.4093
k=1022, #pos=5, #act_iter=99, w_nnz=2897, a_nnz=2437, prod_time=0.17, select_time=0.0385, sub_time=0.223, train_time=1.22

### tron

|pos|: 5
eps = 0.0000000000000000e+00, |g0| = 6.3090098228709794e+03
initial gnorm: 1.32263313e+01, f: 7.03595348e+01
time: 2.7453915216e-02, gnorm: 1.11914703e+01, f: 5.57725424e+01
time: 3.5384926014e-02, gnorm: 1.15318129e+00, f: 5.22716294e+01
time: 4.3441898189e-02, gnorm: 1.33853772e-01, f: 5.21571774e+01
time: 5.1745750010e-02, gnorm: 1.05199506e-02, f: 5.21540136e+01
time: 5.9784205165e-02, gnorm: 9.96152018e-04, f: 5.21539997e+01
time: 6.8109713029e-02, gnorm: 6.95952180e-05, f: 5.21539996e+01
time: 7.6148280408e-02, gnorm: 3.94672185e-06, f: 5.21539996e+01
time: 8.4465249442e-02, gnorm: 2.72058529e-07, f: 5.21539996e+01
time: 9.2742152046e-02, gnorm: 2.55056293e-07, f: 5.21539996e+01
time: 1.0070151929e-01, gnorm: 2.23196951e-07, f: 5.21539996e+01
WARNING: line search fails
num iter: 11
time spent on train one: 0.172081
accuracy test ends


kk: 1386
*********** accuracy test begins ************
randomized CD iter: 200
#pos=16, #ran_iter=200, w_nnz=5001, a_nnz=1361, time=2.38919
optimal obj: 113.532
# iter: 1 time: 0.012461, primal obj: 2581.27
# iter: 2 time: 0.0257144, primal obj: 1722.29
# iter: 3 time: 0.0418269, primal obj: 772.01
# iter: 4 time: 0.0609857, primal obj: 288.951
# iter: 5 time: 0.0838788, primal obj: 161.383
# iter: 6 time: 0.0981659, primal obj: 129.284
# iter: 7 time: 0.111201, primal obj: 118.174
# iter: 8 time: 0.124351, primal obj: 115.703
# iter: 9 time: 0.137859, primal obj: 114.406
# iter: 10 time: 0.15084, primal obj: 113.892
# iter: 11 time: 0.16398, primal obj: 113.721
# iter: 12 time: 0.177231, primal obj: 113.601
# iter: 13 time: 0.190209, primal obj: 113.563
# iter: 14 time: 0.203183, primal obj: 113.55
# iter: 15 time: 0.216101, primal obj: 113.546
# iter: 16 time: 0.229036, primal obj: 113.544
# iter: 17 time: 0.241837, primal obj: 113.542
# iter: 18 time: 0.254636, primal obj: 113.541
# iter: 19 time: 0.267393, primal obj: 113.541
# iter: 20 time: 0.280209, primal obj: 113.541
# iter: 21 time: 0.292983, primal obj: 113.541
# iter: 22 time: 0.305929, primal obj: 113.541
# iter: 23 time: 0.318765, primal obj: 113.541
# iter: 24 time: 0.331616, primal obj: 113.541
# iter: 25 time: 0.344661, primal obj: 113.541
# iter: 26 time: 0.357349, primal obj: 113.541
# iter: 27 time: 0.370206, primal obj: 113.541
# iter: 28 time: 0.383082, primal obj: 113.541
# iter: 29 time: 0.395523, primal obj: 113.541
# iter: 30 time: 0.408202, primal obj: 113.541
# iter: 31 time: 0.420825, primal obj: 113.541
# iter: 32 time: 0.433494, primal obj: 113.541
# iter: 33 time: 0.44664, primal obj: 113.541
# iter: 34 time: 0.459465, primal obj: 113.541
# iter: 35 time: 0.472357, primal obj: 113.541
# iter: 36 time: 0.484967, primal obj: 113.541
# iter: 37 time: 0.49772, primal obj: 113.541
# iter: 38 time: 0.510521, primal obj: 113.541
# iter: 39 time: 0.523211, primal obj: 113.541
# iter: 40 time: 0.535672, primal obj: 113.541
# iter: 41 time: 0.548335, primal obj: 113.541
# iter: 42 time: 0.561043, primal obj: 113.541
# iter: 43 time: 0.573723, primal obj: 113.541
# iter: 44 time: 0.586159, primal obj: 113.541
# iter: 45 time: 0.59894, primal obj: 113.541
# iter: 46 time: 0.611691, primal obj: 113.541
# iter: 47 time: 0.624549, primal obj: 113.541
# iter: 48 time: 0.63729, primal obj: 113.541
# iter: 49 time: 0.649919, primal obj: 113.541
# iter: 50 time: 0.662313, primal obj: 113.541
# iter: 51 time: 0.674849, primal obj: 113.541
# iter: 52 time: 0.687108, primal obj: 113.541
# iter: 53 time: 0.699311, primal obj: 113.541
# iter: 54 time: 0.711581, primal obj: 113.541
# iter: 55 time: 0.723813, primal obj: 113.541
# iter: 56 time: 0.736063, primal obj: 113.541
# iter: 57 time: 0.748253, primal obj: 113.541
# iter: 58 time: 0.760457, primal obj: 113.541
# iter: 59 time: 0.772709, primal obj: 113.541
# iter: 60 time: 0.785005, primal obj: 113.541
# iter: 61 time: 0.79725, primal obj: 113.541
# iter: 62 time: 0.809558, primal obj: 113.541
# iter: 63 time: 0.821845, primal obj: 113.541
# iter: 64 time: 0.83428, primal obj: 113.541
# iter: 65 time: 0.846354, primal obj: 113.541
# iter: 66 time: 0.85849, primal obj: 113.541
# iter: 67 time: 0.870773, primal obj: 113.541
# iter: 68 time: 0.883363, primal obj: 113.541
# iter: 69 time: 0.895735, primal obj: 113.541
# iter: 70 time: 0.908233, primal obj: 113.541
# iter: 71 time: 0.92049, primal obj: 113.541
# iter: 72 time: 0.93268, primal obj: 113.541
# iter: 73 time: 0.944922, primal obj: 113.541
# iter: 74 time: 0.957138, primal obj: 113.541
# iter: 75 time: 0.969346, primal obj: 113.541
# iter: 76 time: 0.981436, primal obj: 113.541
# iter: 77 time: 0.993721, primal obj: 113.541
# iter: 78 time: 1.00569, primal obj: 113.541
# iter: 79 time: 1.01788, primal obj: 113.541
# iter: 80 time: 1.02992, primal obj: 113.541
# iter: 81 time: 1.04198, primal obj: 113.541
# iter: 82 time: 1.05403, primal obj: 113.541
# iter: 83 time: 1.06609, primal obj: 113.541
# iter: 84 time: 1.07801, primal obj: 113.541
# iter: 85 time: 1.09012, primal obj: 113.541
# iter: 86 time: 1.10222, primal obj: 113.541
# iter: 87 time: 1.11435, primal obj: 113.541
# iter: 88 time: 1.12645, primal obj: 113.541
# iter: 89 time: 1.13847, primal obj: 113.541
# iter: 90 time: 1.15075, primal obj: 113.541
# iter: 91 time: 1.16291, primal obj: 113.541
# iter: 92 time: 1.17518, primal obj: 113.541
# iter: 93 time: 1.18744, primal obj: 113.541
# iter: 94 time: 1.19928, primal obj: 113.541
# iter: 95 time: 1.21132, primal obj: 113.541
# iter: 96 time: 1.22347, primal obj: 113.541
# iter: 97 time: 1.2355, primal obj: 113.541
# iter: 98 time: 1.24749, primal obj: 113.541
# iter: 99 time: 1.25964, primal obj: 113.541
k=1386, #pos=16, #act_iter=99, w_nnz=3270, a_nnz=2716, prod_time=0.175, select_time=0.0334, sub_time=0.272, train_time=1.26
*********** accuracy test ends   ************


|pos|: 16
eps = 0.0000000000000000e+00, |g0| = 6.2989101198187564e+03
initial gnorm: 3.57224275e+01, f: 1.47422330e+02
time: 2.9113568831e-02, gnorm: 4.96943804e+01, f: 9.46451800e+01
time: 3.8511817809e-02, gnorm: 9.13652798e+00, f: 7.33055578e+01
time: 4.7544029076e-02, gnorm: 1.51417663e+00, f: 7.10782370e+01
time: 5.7001078967e-02, gnorm: 1.87144889e-01, f: 7.09106979e+01
time: 6.6698516253e-02, gnorm: 2.09637283e-02, f: 7.09059986e+01
time: 7.6404889114e-02, gnorm: 1.53372831e-03, f: 7.09059140e+01
time: 8.5760265123e-02, gnorm: 1.43006454e-04, f: 7.09059137e+01
time: 9.5105121844e-02, gnorm: 1.02103720e-05, f: 7.09059137e+01
time: 1.0410156893e-01, gnorm: 9.55333802e-07, f: 7.09059137e+01
time: 1.1379319197e-01, gnorm: 7.50645127e-08, f: 7.09059137e+01
time: 1.2281390512e-01, gnorm: 4.00796126e-09, f: 7.09059137e+01
WARNING: line search fails
num iter: 12
time spent on train one: 0.208205
accuracy test ends!




### Amazon13k

kk: 1137
*********** accuracy test begins ************
randomized CD iter: 200
#pos=69, #ran_iter=200, w_nnz=203860, a_nnz=30705, time=151.322
optimal obj: 3865.82
# iter: 1 time: 0.49881, primal obj: 368947
# iter: 2 time: 0.92785, primal obj: 308357
# iter: 3 time: 1.37437, primal obj: 244322
# iter: 4 time: 1.84585, primal obj: 191040
# iter: 5 time: 2.31853, primal obj: 139752
# iter: 6 time: 2.81799, primal obj: 103223
# iter: 7 time: 3.34101, primal obj: 67647.6
# iter: 8 time: 3.88975, primal obj: 45701
# iter: 9 time: 4.49379, primal obj: 32474
# iter: 10 time: 5.07619, primal obj: 19501.7
# iter: 11 time: 5.67265, primal obj: 13527.2
# iter: 12 time: 6.29108, primal obj: 9733.05
# iter: 13 time: 6.90971, primal obj: 7133.79
# iter: 14 time: 7.57208, primal obj: 5745.22
# iter: 15 time: 8.26558, primal obj: 5116.75
# iter: 16 time: 8.87151, primal obj: 4749.75
# iter: 17 time: 9.49976, primal obj: 4621.48
# iter: 18 time: 10.1203, primal obj: 4542.28
# iter: 19 time: 10.7699, primal obj: 4461.65
# iter: 20 time: 11.4043, primal obj: 4405.23
# iter: 21 time: 12.0709, primal obj: 4378.27
# iter: 22 time: 12.695, primal obj: 4352.65
# iter: 23 time: 13.3099, primal obj: 4337.8
# iter: 24 time: 13.9363, primal obj: 4318.33
# iter: 25 time: 14.558, primal obj: 4307.41
# iter: 26 time: 15.1871, primal obj: 4291.31
# iter: 27 time: 15.8513, primal obj: 4282.26
# iter: 28 time: 16.475, primal obj: 4276.41
# iter: 29 time: 17.0956, primal obj: 4271.28
# iter: 30 time: 17.717, primal obj: 4268.43
# iter: 31 time: 18.3191, primal obj: 4264.6
# iter: 32 time: 18.9351, primal obj: 4260.34
# iter: 33 time: 19.5946, primal obj: 4258.11
# iter: 34 time: 20.2148, primal obj: 4253.44
# iter: 35 time: 20.843, primal obj: 4249.29
# iter: 36 time: 21.4675, primal obj: 4245.16
# iter: 37 time: 22.0776, primal obj: 4244.3
# iter: 38 time: 22.699, primal obj: 4238.33
# iter: 39 time: 23.3741, primal obj: 4237.22
# iter: 40 time: 24.004, primal obj: 4230.69
# iter: 41 time: 24.615, primal obj: 4229.28
# iter: 42 time: 25.2266, primal obj: 4226.65
# iter: 43 time: 25.8225, primal obj: 4225.71
# iter: 44 time: 26.4502, primal obj: 4225.45
# iter: 45 time: 27.0979, primal obj: 4224.38
# iter: 46 time: 27.7041, primal obj: 4221.83
# iter: 47 time: 28.3212, primal obj: 4220.53
# iter: 48 time: 28.9518, primal obj: 4217.82
# iter: 49 time: 29.5967, primal obj: 4216.37
# iter: 50 time: 30.2325, primal obj: 4214.1
# iter: 51 time: 30.858, primal obj: 4213.4
# iter: 52 time: 31.4776, primal obj: 4213.05
# iter: 53 time: 32.0958, primal obj: 4212.04
# iter: 54 time: 32.7339, primal obj: 4210
# iter: 55 time: 33.3521, primal obj: 4200.27
# iter: 56 time: 33.9726, primal obj: 4198
# iter: 57 time: 34.6169, primal obj: 4196.51
# iter: 58 time: 35.234, primal obj: 4195.95
# iter: 59 time: 35.8735, primal obj: 4195.99
# iter: 60 time: 36.5062, primal obj: 4195.32
# iter: 61 time: 37.1226, primal obj: 4193.56
# iter: 62 time: 37.7385, primal obj: 4193.24
# iter: 63 time: 38.3661, primal obj: 4193.07
# iter: 64 time: 39.0008, primal obj: 4192.34
# iter: 65 time: 39.6341, primal obj: 4191.21
# iter: 66 time: 40.2535, primal obj: 4188.19
# iter: 67 time: 40.8777, primal obj: 4187.32
# iter: 68 time: 41.4952, primal obj: 4186.54
# iter: 69 time: 42.1158, primal obj: 4185.38
# iter: 70 time: 42.7445, primal obj: 4184.53
# iter: 71 time: 43.3818, primal obj: 4183.81
# iter: 72 time: 44.0197, primal obj: 4183.16
# iter: 73 time: 44.6507, primal obj: 4180.67
# iter: 74 time: 45.287, primal obj: 4178.65
# iter: 75 time: 45.9194, primal obj: 4176
# iter: 76 time: 46.5765, primal obj: 4175.83
# iter: 77 time: 47.2082, primal obj: 4175.78
# iter: 78 time: 47.8244, primal obj: 4175.59
# iter: 79 time: 48.4613, primal obj: 4174.22
# iter: 80 time: 49.0701, primal obj: 4173.37
# iter: 81 time: 49.6867, primal obj: 4173.38
# iter: 82 time: 50.3348, primal obj: 4173.19
# iter: 83 time: 50.9586, primal obj: 4172.05
# iter: 84 time: 51.6263, primal obj: 4171.12
# iter: 85 time: 52.2594, primal obj: 4169.58
# iter: 86 time: 52.8827, primal obj: 4168.55
# iter: 87 time: 53.5277, primal obj: 4168.36
# iter: 88 time: 54.1543, primal obj: 4168.18
# iter: 89 time: 54.7712, primal obj: 4167.75
# iter: 90 time: 55.3939, primal obj: 4167.07
# iter: 91 time: 56.0792, primal obj: 4166.7
# iter: 92 time: 56.6925, primal obj: 4165.48
# iter: 93 time: 57.3358, primal obj: 4164.54
# iter: 94 time: 57.942, primal obj: 4164.61
# iter: 95 time: 58.5719, primal obj: 4164.39
# iter: 96 time: 59.1969, primal obj: 4163.69
# iter: 97 time: 59.8154, primal obj: 4163.33
# iter: 98 time: 60.4431, primal obj: 4162.67
# iter: 99 time: 61.0622, primal obj: 4162.33
k=1137, #pos=69, #act_iter=99, w_nnz=46125, a_nnz=60028, prod_time=20.1, select_time=1.75, sub_time=3.48, train_time=61.1

|pos|: 69
eps = 0.0000000000000000e+00, |g0| = 1.6582196421538005e+05
initial gnorm: 1.38922817e+02, f: 4.01897868e+03
time: 1.1157285529e+00, gnorm: 1.13479802e+02, f: 3.11585505e+03
time: 1.5012099389e+00, gnorm: 2.17607011e+01, f: 2.89679392e+03
time: 1.8535901620e+00, gnorm: 5.25891325e+01, f: 2.89513742e+03
time: 2.1126257810e+00, gnorm: 2.62436729e+00, f: 2.88229098e+03
time: 2.4764963551e+00, gnorm: 4.45071571e+00, f: 2.88187558e+03
time: 2.8116020341e+00, gnorm: 3.73622641e-01, f: 2.88154958e+03
time: 3.1819864949e+00, gnorm: 3.99867423e-01, f: 2.88154429e+03
time: 3.5378507669e+00, gnorm: 3.71760388e-02, f: 2.88153953e+03
time: 3.9150532968e+00, gnorm: 1.94870869e-03, f: 2.88153948e+03
time: 4.2822938608e+00, gnorm: 1.59717152e-04, f: 2.88153948e+03
time: 4.6601375178e+00, gnorm: 1.47464292e-05, f: 2.88153948e+03
time: 5.0090118838e+00, gnorm: 1.68161403e-06, f: 2.88153948e+03
time: 5.3773268508e+00, gnorm: 8.47184767e-08, f: 2.88153948e+03
time: 5.7612787420e+00, gnorm: 8.40566295e-08, f: 2.88153948e+03
time: 6.1432089298e+00, gnorm: 8.30247544e-09, f: 2.88153948e+03
time: 6.5109690372e+00, gnorm: 5.96241364e-10, f: 2.88153948e+03
time: 6.8574499418e+00, gnorm: 2.99290706e-10, f: 2.88153948e+03
time: 7.2594196419e+00, gnorm: 2.98705373e-10, f: 2.88153948e+03
WARNING: line search fails
num iter: 19
time spent on train one: 14.415320
accuracy test ends!




kk: 9673
*********** accuracy test begins ************
randomized CD iter: 200
#pos=54, #ran_iter=200, w_nnz=203860, a_nnz=28856, time=150.02
optimal obj: 3551.83
# iter: 1 time: 0.419325, primal obj: 354179
# iter: 2 time: 0.862653, primal obj: 330962
# iter: 3 time: 1.27899, primal obj: 256725
# iter: 4 time: 1.72838, primal obj: 207343
# iter: 5 time: 2.19955, primal obj: 169074
# iter: 6 time: 2.69253, primal obj: 132219
# iter: 7 time: 3.17673, primal obj: 86974.5
# iter: 8 time: 3.74413, primal obj: 65873.1
# iter: 9 time: 4.27768, primal obj: 50393.2
# iter: 10 time: 4.81747, primal obj: 30279.2
# iter: 11 time: 5.38708, primal obj: 20737.5
# iter: 12 time: 5.96539, primal obj: 13410.4
# iter: 13 time: 6.60137, primal obj: 9048.2
# iter: 14 time: 7.28, primal obj: 6967.96
# iter: 15 time: 7.90283, primal obj: 5526.84
# iter: 16 time: 8.54416, primal obj: 4812.2
# iter: 17 time: 9.15772, primal obj: 4451.72
# iter: 18 time: 9.75643, primal obj: 4311.57
# iter: 19 time: 10.3713, primal obj: 4206.71
# iter: 20 time: 11.0213, primal obj: 4146.67
# iter: 21 time: 11.6284, primal obj: 4111.08
# iter: 22 time: 12.2176, primal obj: 4082.09
# iter: 23 time: 12.8281, primal obj: 4064.25
# iter: 24 time: 13.4398, primal obj: 4041.7
# iter: 25 time: 14.0395, primal obj: 4018.18
# iter: 26 time: 14.6648, primal obj: 4010.04
# iter: 27 time: 15.2861, primal obj: 3995.59
# iter: 28 time: 15.8978, primal obj: 3985.31
# iter: 29 time: 16.5138, primal obj: 3974.63
# iter: 30 time: 17.1228, primal obj: 3967.08
# iter: 31 time: 17.7163, primal obj: 3963.38
# iter: 32 time: 18.3227, primal obj: 3961.13
# iter: 33 time: 18.9332, primal obj: 3958.57
# iter: 34 time: 19.5562, primal obj: 3956.63
# iter: 35 time: 20.1567, primal obj: 3950.48
# iter: 36 time: 20.7587, primal obj: 3947.58
# iter: 37 time: 21.3745, primal obj: 3942.22
# iter: 38 time: 21.9838, primal obj: 3937.15
# iter: 39 time: 22.5981, primal obj: 3933.36
# iter: 40 time: 23.2162, primal obj: 3932.42
# iter: 41 time: 23.8529, primal obj: 3931.4
# iter: 42 time: 24.4535, primal obj: 3924.75
# iter: 43 time: 25.0645, primal obj: 3924
# iter: 44 time: 25.678, primal obj: 3922.2
# iter: 45 time: 26.2771, primal obj: 3918.64
# iter: 46 time: 26.8965, primal obj: 3918.3
# iter: 47 time: 27.514, primal obj: 3917.34
# iter: 48 time: 28.1356, primal obj: 3913.39
# iter: 49 time: 28.7331, primal obj: 3910.99
# iter: 50 time: 29.34, primal obj: 3908.04
# iter: 51 time: 29.9548, primal obj: 3905.68
# iter: 52 time: 30.5898, primal obj: 3905.1
# iter: 53 time: 31.2053, primal obj: 3903.62
# iter: 54 time: 31.8342, primal obj: 3902.51
# iter: 55 time: 32.4455, primal obj: 3899.39
# iter: 56 time: 33.0572, primal obj: 3896.47
# iter: 57 time: 33.6701, primal obj: 3893.63
# iter: 58 time: 34.2833, primal obj: 3892.82
# iter: 59 time: 34.8883, primal obj: 3891.2
# iter: 60 time: 35.4957, primal obj: 3890.48
# iter: 61 time: 36.1229, primal obj: 3890
# iter: 62 time: 36.7256, primal obj: 3887.25
# iter: 63 time: 37.3432, primal obj: 3887.05
# iter: 64 time: 37.9654, primal obj: 3886.92
# iter: 65 time: 38.5887, primal obj: 3885.53
# iter: 66 time: 39.2005, primal obj: 3885.2
# iter: 67 time: 39.8254, primal obj: 3884.07
# iter: 68 time: 40.4479, primal obj: 3882.62
# iter: 69 time: 41.0537, primal obj: 3881.74
# iter: 70 time: 41.6676, primal obj: 3881.68
# iter: 71 time: 42.2749, primal obj: 3881.23
# iter: 72 time: 42.8747, primal obj: 3880.38
# iter: 73 time: 43.4788, primal obj: 3879.93
# iter: 74 time: 44.077, primal obj: 3879.29
# iter: 75 time: 44.6867, primal obj: 3879.1
# iter: 76 time: 45.2931, primal obj: 3879.1
# iter: 77 time: 45.9182, primal obj: 3878.11
# iter: 78 time: 46.5427, primal obj: 3878.05
# iter: 79 time: 47.1574, primal obj: 3878.05
# iter: 80 time: 47.7603, primal obj: 3877.98
# iter: 81 time: 48.3573, primal obj: 3877.97
# iter: 82 time: 48.9722, primal obj: 3876.35
# iter: 83 time: 49.6293, primal obj: 3876.19
# iter: 84 time: 50.2496, primal obj: 3876.19
# iter: 85 time: 50.8545, primal obj: 3876.19
# iter: 86 time: 51.4714, primal obj: 3875.87
# iter: 87 time: 52.0891, primal obj: 3875.93
# iter: 88 time: 52.7014, primal obj: 3874.99
# iter: 89 time: 53.3385, primal obj: 3874.77
# iter: 90 time: 53.9657, primal obj: 3874.76
# iter: 91 time: 54.5757, primal obj: 3873.93
# iter: 92 time: 55.1861, primal obj: 3873.82
# iter: 93 time: 55.8059, primal obj: 3873.83
# iter: 94 time: 56.4173, primal obj: 3872.11
# iter: 95 time: 57.0601, primal obj: 3871.21
# iter: 96 time: 57.6572, primal obj: 3871.01
# iter: 97 time: 58.2641, primal obj: 3871
# iter: 98 time: 58.8669, primal obj: 3870.94
# iter: 99 time: 59.4819, primal obj: 3870.91
k=9673, #pos=54, #act_iter=99, w_nnz=43069, a_nnz=56259, prod_time=19.7, select_time=1.72, sub_time=3.06, train_time=59.5


|pos|: 54
eps = 0.0000000000000000e+00, |g0| = 1.6583187569510279e+05
initial gnorm: 1.16497756e+02, f: 3.24785078e+03
time: 1.0782989077e+00, gnorm: 7.53125010e+01, f: 2.75085841e+03
time: 1.4476117161e+00, gnorm: 2.89537225e+01, f: 2.63505019e+03
time: 1.7621922251e+00, gnorm: 9.76232539e+00, f: 2.62325961e+03
time: 2.0896073729e+00, gnorm: 8.31797336e+00, f: 2.62275413e+03
time: 2.3518599588e+00, gnorm: 8.27720471e-01, f: 2.62204036e+03
time: 2.6997712208e+00, gnorm: 9.17770730e-02, f: 2.62192744e+03
time: 3.0477533648e+00, gnorm: 6.89582100e-03, f: 2.62192664e+03
time: 3.4043400637e+00, gnorm: 5.59074465e-04, f: 2.62192664e+03
time: 3.7436968358e+00, gnorm: 4.80834528e-05, f: 2.62192664e+03
time: 4.0916038938e+00, gnorm: 4.62024412e-06, f: 2.62192664e+03
time: 4.4193005557e+00, gnorm: 2.32081500e-06, f: 2.62192664e+03
time: 4.8450375469e+00, gnorm: 2.32079730e-06, f: 2.62192664e+03
time: 5.2822451121e+00, gnorm: 2.32079287e-06, f: 2.62192664e+03
WARNING: line search fails
num iter: 14
time spent on train one: 11.580397
accuracy test ends!



./util/a.out 9999999 20000000

717032598

350000000

400000000

2147483647


number of subtrees: 496573
height of the whole tree: 33
