#!/usr/bin/env python
import sys
import os

from Engine import Engine
from Config import Config
from Log import log
import tensorflow as tf
import numpy
from negative_actor import *
from negative_critic import *
from positive_actor import *
from positive_critic import *
from new_vgg19 import *
from parameter import *


def init_log(config):
  log_dir = config.dir("log_dir", "logs")
  model = config.unicode("model")
  filename = log_dir + model + ".log"
  verbosity = config.int("log_verbosity", 3)
  log.initialize([filename], [verbosity], [])


def main(_):

<<<<<<< HEAD
  numpy.random.seed(300)
  tf.set_random_seed(300)
=======
  #numpy.random.seed(300)
  #tf.set_random_seed(300)
>>>>>>> 1f482adf942d5479b99a739478b3da6afcc1386a

  assert len(sys.argv) == 2, "usage: main.py <config>"
  config_path = sys.argv[1]
  assert os.path.exists(config_path), config_path
  try:
    config = Config(config_path)
  except ValueError as e:
    print "Malformed config file:", e
    return -1
  init_log(config)
  config.initialize()
  #dump the config into the log
  print >> log.v4, open(config_path).read()

  dataset_path = DATSET_PATH
  #label_path = LABEL_PATH

  with open(NAME_SEQ, 'r') as f:
    name_seqs = f.read().splitlines()
  num_seqs = len(name_seqs)

  path_prefix = "/home/smj/DataSet/DAVIS/DAVIS"

  f = open("./pair", "r")
  records = f.read().splitlines()

  img_path_seqs = []

  for i in range(num_seqs):
    img_path_seqs.append([])
    for j in range(TOTAL_FRAME):
      path = records[0].split()[0]
      img_path_seqs[i].append(path_prefix + path)
      records.pop(0)

  vgg = VGG19()
  vgg.restore()

  negative_actor = Negative_Actor(4, LR_A)
  negative_critic = Negative_Critic(LR_C)

  positive_actor = Positive_Actor(2, LR_A)
  positive_critic = Positive_Critic(LR_C)

  if FIRST_TRAIN == True:
      negative_actor.initialze()
      negative_actor.save()
      negative_critic.initialze()
      negative_critic.save()
      positive_actor.initialze()
      positive_actor.save()
      positive_critic.initialze()
      positive_critic.save()

  negative_actor.restore()
  negative_critic.restore()
  positive_actor.restore()
  positive_critic.restore()

  engine = Engine(config, vgg, negative_actor, negative_critic, positive_actor, positive_critic, img_path_seqs, name_seqs)
  engine.run()

if __name__ == '__main__':
  tf.app.run(main)
