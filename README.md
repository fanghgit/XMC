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
  $ python util/remap.py data/eurlex/train.txt data/eurlex/test.txt data/eurlex/train_remap.txt data/eurlex/test_remap.txt
  ```
Do if-idf transformation:
  ```
  $ python util/transform.py data/eurlex/train_remap.txt data/eurlex/test_remap.txt data/eurlex/train_remap_tfidf.txt data/eurlex/test_remap_tfidf.txt
  ```
Go to linesearch folder and type `make` to build executable

Training with default initialization for EURLex:
  ```
  $ ./linesearch/train -s 2 -e 0.0001 data/eurlex/train_remap_tfidf.txt
linesearch/models/eu1.model
  ```
Training with proposed initialization for EURLex:
  ```
  $ ./linesearch/train -s 2 -m 1 -e 0.0001 data/eurlex/train_remap_tfidf.txt linesearch/models/eu2.model
  ```
Note the we changed the stopping criterion for L2L2 solver, in
liblinear, we stop when 

  |f'(w)|_2 <= eps\*min(pos,neg)/l\*|f'(w0)|_2,
  where f initializations the primal function and pos/neg are # of
  positive/negative data (Descriptionfault 0.01)

Here we stop when 

  |f'(w)|_2 <= eps|f'(w0)|_2, we can set eps = 0.001 or 0.0001 for experiments

DISMEC's code is changed to use the same stopping criterion.

Prediction: (Same as DISMEC)

process data
  ```
  $ javac util/LabelExtractor.java
  ```
  ```
  $ java util/LabelExtractor data/eurlex/test_remap_tfidf.txt data/eurlex/test_remap_tfidf_zeroed.txt data/eurlex/GS.txt
  ```
Produce output:
  ```
  $ linesearch/predict data/eurlex/test_remap_tfidf_zeroed.txt linesearch/models/eu2.model linesearch/output/eu.out
  ```
Performance evaluation
  ```
  $ javac util/DistributedPredictor.java
  ```
  ```
  $ java util/DistributedPredictor linesearch/output linesearch/final-output/top1.out linesearch/final-output/top3.out linesearch/final-output/top5.out
  ```
Calculate Precision and NDCG:
  ```
  $ javac util/MultiLabelMetrics.java
  ```
  ```
  $ java util/MultiLabelMetrics.java data/eurlex/GS.txt linesearch/final-output/top1.out linesearch/final-output/top3.out linesearch/final-output/top5.out
  ```

Delete `top1.out` .. before evalute another model. 


