# -*- coding: utf-8 -*-

import csv
import glob
import time
import queue
import struct
import numpy as np
import tensorflow as tf
from random import shuffle
from threading import Thread
from tensorflow.core.example import example_pb2

from utils import utils
from utils import config

import random
random.seed(1234)

logger = tf.get_logger()




