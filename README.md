# Experimental Code for XMC
## Description
This repo currently includes the following folders:
- data: folder to hold experimental data, currently only have EURLex
- util: contains code for preprocessing and evalution (some are from
  DISMEC)
- dismec: DISMEC's code
- linesearch: liblinear whose L2L2 is solved by Newton method with line
  search. linesearch in primary built on the code of [3]

## Quick Start
If the data is download from XMC website, remove the first line.

Run the following code to remap features and labels.
  ```
  $ python util/remap.py data/eurlex/train.txt data/eurlex/test.txt data/eurlex/train_remap.txt data/eurlex/test_remap.txt -r 1
  ```
Do if-idf transformation:
  ```
  $ python util/transform.py data/eurlex/train_remap.txt data/eurlex/test_remap.txt data/eurlex/train_remap_tfidf.txt data/eurlex/test_remap_tfidf.txt
  ```
Go to linesearch_parallel folder and type `make` to build executable.

Create folder to hold models: `mkdir linesearch_parallel/models`

Parameter options:
  ```
    -s : type of solver, default to be L2L2 primal solver
        -0: L2R_LR
        -1: L2R_L2LOSS_SVC
        -2: L1R_L2LOSS_SVC
        -3: L1R_LR
        (default 1)
    -x : initialization
        -0: naive 0 initialization
        -1: all negative initialization
        -2: MST initialization
        (default 0)
    -B : bias
        (default 0)
    -P : number of threads
        (default 1)
    -e : stopping criterion
        |f'(w)|_2 <= min(0.0001, eps*min(pos,neg)/l ) *|f'(0)|_2
        (default 1.0)
  ```
Training with default initialization for EURLex:
  ```
  $ ./linesearch_parallel/train -B 1 -P 1 -e 1.0 0.0001 data/eurlex/train_remap_tfidf.txt linesearch_parallel/models/eu1.model
  ```
Training with all negative initialization for EURLex:
  ```
  $ ./linesearch_parallel/train -B 1 -P 1 -x 1 -e 1.0 0.0001 data/eurlex/train_remap_tfidf.txt linesearch_parallel/models/eu2.model
  ```
Training with MST initialization for EURLex:
  ```
  $ ./linesearch_parallel/train -B 1 -P 1 -x 2 -e 1.0 0.0001 data/eurlex/train_remap_tfidf.txt linesearch_parallel/models/eu2.model
  ```

By using `-x 1` or `-x 2`, training should be able to finish in around 200 sec.

**Remark:** `-B 1` or `-B 0.1` will not change prediction accuracy, but it will affect model size significantly, `-B 0.1` will make our algorithm faster since the # of iterations relies on $$\|x\|_2$$, but `-B 0.1` will increase the model size.


**Stopping criterion**

We stop when

``|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2``

and

``|f'(w)|_2 <= 0.0001*|f'(w0)|_2``

where `f` initializations the primal function and `w0 = zeros(n,1)` and pos/neg are # ofpositive/negative data. pos/neg is very small for most classes, so we set `eps = 1.0` or `eps=0.1` for experiments.


DISMEC's code is changed to use the same stopping criterion.

Prediction: create new folder `mkdir linesearch_parallel/output`

  ```
  $ ./linesearch_parallel/predict data/eurlex/test_remap_tfidf.txt linesearch_parallel/models/eu2.model linesearch_parallel/output/eu2.out
  ```

Training with DISMEC:
  ```
  $ ./dismec/dismec/train -B 1 -s 2 -e 0.0001 data/eurlex/train_remap_tfidf.txt dismec/dismec/models/eu_dismec.model
  ```
  ```
  $ ./dismec/dismec/predict data/eurlex/test_remap_tfidf_zeroed.txt dismec/dismec/models/eu_dismec.model dismec/dismec/output/eu_dismec.out
  ```
  ```
  $ ./python util/evaluate.py data/eurlex/GS.txt dismec/dismec/output/eu_dismec.out
  ```
## References:

[1] Rohit Babbar, Bernhard Shoelkopf, DiSMEC - Distributed Sparse Machines for Extreme Multi-label Classification, 2017

[2] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. LIBLINEAR: A library for large linear classification, 2008

[3] C.-Y. Hsia, Y. Zhu, and C.-J. Lin. A study on trust region update rules in Newton methods for large-scale linear classification, 2017
