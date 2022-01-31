# General tips for doing NLP / DL research

## Finding existing research

Often helpful to check existing research how they do the research, e.g., what task they used, what datasets are common in the area, which evaluation metrics to use.  Reading papers will save you A LOT of time
- Browse recent publications at any of the top venues where NLP / Deep learning is published: ACL, EMNLP, TACL, NAACL, EACL, NIPS, ICLR, ICML, NIPS, etc.
- Publications at many NLP venues are indexed at http://www.aclweb.org/anthology

## Find datasets

We generally DO NOT recommend collecting your own datasets because it is very time consuming and prone to messy process.  Also, it does not allow comparison with past works.  Try use existing datasets, and instead focus on improving and understanding the models

- check out https://nlpprogress.com
- check out https://machinelearningmastery.com/datasets-natural-language-processing
- check out https://github.com/niderhoff/nlp-datasets
- check out http://statmt.org
- check out http://huggingface.co/datasets

And much more!

## Exciting areas 2021
- Robstness to domain shift or adversarial attacks
- Doing analytical work what, why, how large pretrained models learned
- Transfer learning
- Zero-shot learning / Few-shot learning on very small datasets
- Looking at gender bias, trustworthiness, explainability of large models
- Low resource languages (e.g., Somali)
- Scaling models down (e.g., GPT-3 is too big, can we prune it while preserving performances?)
- Establishing evaluation metrics (e.g., BLEU or ROUGE was found very incorrelated with human scores)

## Steps to do NLP research

#### 1. Define goals
- **Clearly** define your task.  What's the input and output?
- What dataset(s) you will use?
- What is your evaluation metric?
- What defines success of your work

#### 2. Data preprocessing
- Use NLTK or spaCy or TorchText to help you preprocess your data
- Read this for full pipeline NLP code: https://github.com/chaklam-silpasuwanchai/Python-for-Data-Science/tree/master/Lectures/04-NLP

#### 3. Data preparation
Correctly splitting is key to avoid overfitting and promote generalization. 

- Split into (1) training, (2) dev / validation, (3) test set, with a ratio of around 90/5/5 (if the dataset is very big)
- Use training set for optimizing the parameters
- Use dev set to compare performance across models; tune hyperparameters
- Use test set for reporting your metric (NEVER ever touch this until the very end)
- Always randomize the order of samples, e.g., no one sentence always comes after another sentence

#### 4. Buid strong baselines
Before attempting to develop more advanced model, prepare yourself a **baseline** model.  I cannot stress enough how important is this.   A  baseline could be Naive Bayes, a simple LSTM/RNN single layer network, or simply using word embeddings. But make sure your baseline is **NOT too weak**....read papers what is the baseline for the area.

#### 5. Training and debugging neural models
- Debugging takes a lot of time if you always use the entire dataset.  That is NOT SO SMART.  At the beginning, use only a small toy dataset (e.g., small fraction of training data, or a hand-created dataset).  This will enable quick debugging and coding.
- Can use Huggingface for pretrained models
- Use regularization and stopping criteria to avoid overfitting
- Use performance on the dev set to tune hyperparameters
- Use ablation experiments, , i.e., remove some part of the full model, to know what goes wrong/right in your model
- Read http://ruder.io/deep-learning-nlp-best-practices/
- Read https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/

#### 6. Evaluation
- Choose a typical evaluation metric, such that it allows comparison between your work and past works
- Human evaluation is often needed, if that evaluation metric is still on progress. For example, in NLP, we are mostly still lacking very good evaluation metric for very open-ended generation such as question-answering, dialogue, summarization.  Thus in this area, complement typical evaluation metric with human judgements.
- Compare your metric with past work 
- Read my Lecture 14: Analysis of Model's Inner Workings to get some idea the possible analysis you can made
- Check out https://gluebenchmark.com/tasks

