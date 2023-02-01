# Pointer-Generator-Pytorch

## About
The pytorch implementation of [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368). 
This implementation also tries to use Transformer as the encoder.
The project are heavily borrowed from [atulkum-pointer_summarizer](https://github.com/atulkum/pointer_summarizer.git) and 
[jadore801120-attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

## Requirements
* python==3.7.4
* pytorch==1.4.0
* pyrouge==0.1.3
* tensorflow>=1.13.1

## Quick start
* The path and parameters of project:
you might need to change some path and parameters in utils/config.py according your setup.
* Dataset:
you can download the CNN/DailyMail dataset from https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail, 
then run make-datafiles.py to process data. For the specific process, you can refer to https://github.com/abisee/cnn-dailymail.
* Run: 
you can run train.py, eval.py, and test for training, evaluating, and test, respectively.

### Note:
* There is only single example repeated across the batch in the decode mode of beam search.


