# Run IR evaluation

First, create a new directory called `result_`.

```
mkdir result_
```

Then make sure `systems` is in the same directory as code does. Run `EVAL.py` 

```
python EVAL.py
```

and `S*.eval` and `All.eval` will show up in `result_`.



# Text Classification

Make sure `tweetsclassification` is in the same directory as code does. It should contain `Tweets.14cat.test`, `Tweets.14cat.train` and `classIDs.txt`.

## Create `feats.dic ` and Convert training and test set to feature vectors

```
opt=rm_punc,lower,stem,stop,hash_at
python classi_advanced.py -m fd,cf -pre_opt $opt
```

`-m: mode`

- fd: create *f*eats.*d*ic
- cf: *c*reate *f*eature vectors

`opt`

- rm_punc: remove punctuation
- lower: casefold
- stem: stemming
- stop: remove stopwords
- hash_at: duplicate hashtags and ats.

This command will generate all files in `tweetsclassification` .

## SVM training

SVM training is the same as that in lab. 

   `svm_multiclass_learn -c 1000 feats.train model`

   `svm_multiclass_classify feats.test model pred.out`

## Naïve Bayes training and test

Make sure create `feat.dic` and training and test set are converted into feature vectors.

```
python train_and_pred.py -method bayes -mode test,train
```

It will generate `model_bayes.pkl` and `pred_bayes.out` in `tweetsclassification` .

## Evaluate prediction

```
python eval_textclassi.py -gt tweetsclassification/feats.test -pred tweetsclassification/pred.out
```

It will print the result on terminal.

