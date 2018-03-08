# Experimental Code for XMC
## Description
This repo currently includes the following folders:
- data: folder to hold experimental data, currently only have EURLex
- util: contains code for preprocessing and evalution (some are from
  DISMEC)
- dismec: DISMEC's code
- linesearch: liblinear whose L2L2 is solved by Newton method with line
  search.

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
Go to linesearch folder and type `make` to build executable(will generate many warnings now)

Create folder to hold models: `mkdir linesearch/models`

Training with default initialization for EURLex:
  ```
  $ ./linesearch/train -s 2 -e 0.0001 data/eurlex/train_remap_tfidf.txt linesearch/models/eu1.model
  ```
Training with proposed initialization for EURLex:
  ```
  $ ./linesearch/train -s 2 -m 1 -e 0.0001 data/eurlex/train_remap_tfidf.txt linesearch/models/eu2.model
  ```
Should be able to finish in around 200 sec.

Note the we changed the stopping criterion for L2L2 solver, in
liblinear, we stop when 

  |f'(w)|_2 <= eps\*min(pos,neg)/l\*|f'(w0)|_2,
  where f initializations the primal function and pos/neg are # of
  positive/negative data (Descriptionfault 0.01)

Here we stop when 

  |f'(w)|_2 <= eps|f'(w0)|_2, we can set eps = 0.001 or 0.0001 for experiments

DISMEC's code is changed to use the same stopping criterion.

Prediction: (Same as DISMEC)

create new folders: `mkdir linesearch/output`

Process test data, split labels from test data
  ```
  $ python util/splitlabel.py data/eurlex/test_remap_tfidf.txt data/eurlex/GS.txt data/eurlex/test_remap_tfidf_zeroed.txt
  ```
Produce output:
  ```
  $ linesearch/predict data/eurlex/test_remap_tfidf_zeroed.txt linesearch/models/eu2.model linesearch/output/eu2.out
  ```
Performance evaluation
  ```
  $ python util/evaluate.py data/eurlex/GS.txt linesearch/output/eu2.out
  ```

Training with DISMEC:
  ```
  ./dismec/dismec/train -s 2 -e 0.0001 data/eurlex/train_remap_tfidf.txt dismec/dismec/models/eu_dismec.model
  ```
  ```
  dismec/dismec/predict data/eurlex/test_remap_tfidf_zeroed.txt dismec/dismec/models/eu_dismec.model dismec/dismec/output/eu_dismec.out
  ```
  ```
  python util/evaluate.py data/eurlex/GS.txt dismec/dismec/output/eu_dismec.out
  ```

