# this python file is used to train the RL model where the state is only the original image

import os
import cv2
import numpy as np
import os
import copy
from skimage import transform
import time
from parameter import *
from actor import *
from critic import *
from new_vgg19 import *
import random

# help function

def get_state(current_path):
  # import the image of the current frame
  img_path = os.path.join(current_path, "img_t.jpg")
  img = cv2.imread(img_path)

  # generate the average number of BGR
  b_1d = np.full((480, 854, 1), 104)
  g_1d = np.full((480, 854, 1), 117)
  r_1d = np.full((480, 854, 1), 124)
  bgr_mean = np.concatenate((b_1d, g_1d), axis=2)
  bgr_mean = np.concatenate((bgr_mean, r_1d), axis=2)

  # generate the image minused by the average value
  #img = img - bgr_mean

  # import the probability map of the current map
  # (480, 854), the probability that this pixel belongs to the foreground
  probability_map_path = os.path.join(current_path, "map.npy")
  probability_map = np.load(probability_map_path)
  probability_map = probability_map[:, :, 1]

  # generate the candidate adpatation areas with different threshold before cropping
  area_97 = probability_map > 0.97
  area_70 = probability_map > 0.7
  area_40 = probability_map > 0.4
  area_20 = probability_map > 0.2
  area_10 = probability_map > 0.1

  # generate the bounding box the area_10 (biggest candicate adaptation area)
  mask_bounding = np.full((480, 854), 1) * area_10
  mask_bounding = mask_bounding.astype(np.uint8)
  x, y, w, h = cv2.boundingRect(mask_bounding)

  # crop all candidate adaptation areas and the original image
  img = img[y:y + h, x:x + w]
  area_97 = area_97[y:y + h, x:x + w]
  area_70 = area_70[y:y + h, x:x + w]
  area_40 = area_40[y:y + h, x:x + w]
  area_20 = area_20[y:y + h, x:x + w]
  area_10 = area_10[y:y + h, x:x + w]

  # mask (RGB image) of adaptation area with probability greater than 0.97
  area_97_d1 = area_97.reshape(h, w, 1)
  area_97_d2 = np.concatenate((area_97_d1, area_97_d1), axis=2)
  area_97_d3 = np.concatenate((area_97_d2, area_97_d1), axis=2)
  mask_97 = img * area_97_d3

  # mask (RGB image) of adaptation area with probability greater than 0.7
  area_70_d1 = area_70.reshape(h, w, 1)
  area_70_d2 = np.concatenate((area_70_d1, area_70_d1), axis=2)
  area_70_d3 = np.concatenate((area_70_d2, area_70_d1), axis=2)
  mask_70 = img * area_70_d3

  # mask (RGB image) of adaptation area with probability greater than 0.4
  area_40_d1 = area_40.reshape(h, w, 1)
  area_40_d2 = np.concatenate((area_40_d1, area_40_d1), axis=2)
  area_40_d3 = np.concatenate((area_40_d2, area_40_d1), axis=2)
  mask_40 = img * area_40_d3

  # mask (RGB image) of adaptation area with probability greater than 0.2
  area_20_d1 = area_20.reshape(h, w, 1)
  area_20_d2 = np.concatenate((area_20_d1, area_20_d1), axis=2)
  area_20_d3 = np.concatenate((area_20_d2, area_20_d1), axis=2)
  mask_20 = img * area_20_d3

  # mask (RGB image) of adaptation area with probability greater than 0.1
  area_10_d1 = area_10.reshape(h, w, 1)
  area_10_d2 = np.concatenate((area_10_d1, area_10_d1), axis=2)
  area_10_d3 = np.concatenate((area_10_d2, area_10_d1), axis=2)
  mask_10 = img * area_10_d3

  # resize all masks
  mask_97_resized = transform.resize(mask_97, [224, 224, 3])
  mask_70_resized = transform.resize(mask_70, [224, 224, 3])
  mask_40_resized = transform.resize(mask_40, [224, 224, 3])
  mask_20_resized = transform.resize(mask_20, [224, 224, 3])
  mask_10_resized = transform.resize(mask_10, [224, 224, 3])

  # only for test the current image
  # cv2.imshow("1", mask_10_resized)
  # cv2.waitKey(0)

  # extract the feature of all masks
  mask_97_feature = vgg.extract_conv5(mask_97_resized)
  mask_70_feature = vgg.extract_conv5(mask_70_resized)
  mask_40_feature = vgg.extract_conv5(mask_40_resized)
  mask_20_feature = vgg.extract_conv5(mask_20_resized)
  mask_10_feature = vgg.extract_conv5(mask_10_resized)

  # reshape all feature maps
  mask_97_feature = np.reshape(mask_97_feature, (7, 7, 512))
  mask_70_feature = np.reshape(mask_70_feature, (7, 7, 512))
  mask_40_feature = np.reshape(mask_40_feature, (7, 7, 512))
  mask_20_feature = np.reshape(mask_20_feature, (7, 7, 512))
  mask_10_feature = np.reshape(mask_10_feature, (7, 7, 512))

  state_positive = np.concatenate((mask_97_feature, mask_70_feature), axis=2)
  state_positive = state_positive.reshape(1, 7, 7, 1024)

  state_negative = np.concatenate((mask_40_feature, mask_20_feature, mask_10_feature), axis=2)
  state_negative = state_negative.reshape(1, 7, 7, 1536)

  '''
  img_resized = transform.resize(img, [224, 224, 3])
  state_t = vgg.extract_conv5(img_resized)
  return state_t
  '''

  return state_negative

# constant varibale
ITERATION_NUM = 50000
FIRST_TRAIN = False

# initialize models
vgg = VGG19()
vgg.restore()

actor = Actor(4, LR_A)
critic = Critic(LR_C)

if FIRST_TRAIN == True:
    actor.initialze()
    actor.save()
    critic.initialze()
    critic.save()
    #actor.save_learning_rate()
    #critic.save_learning_rate()
else:
  #actor.restore_learning_rate()
  #critic.restore_learning_rate()

  actor.restore()
  critic.restore()
  print("model restored")

# load the training data

# root path of the training data
root_path = "/home/smj/OnAVOS_new"

# a certain train node indicates one forward result data with the correct threshold,
# like "breakdance1" and "camel2", etc.
train_nodes = os.listdir(root_path)
train_nodes.sort()

# another way set a specific traning node
train_nodes = ["breakdance1", "camel1"]

mean_loss = 0

# external loop for the iterate
# internal loop for different training node (like, breakdance1, breakdance2, bmx-bike1, etc)
for i in range(ITERATION_NUM):

  print()

  for node in train_nodes:
    print(node)

    # list to store each correct action and predicted action
    correct_action_list = []
    a_t_list = []

    # to count the action number
    count = 0
    # to count the number that the predicted action is same to the correct one
    match_count = 0

    current_path = os.path.join(root_path, node)

    # import the image of the current frame
    img_path = os.path.join(current_path, "img_t.jpg")
    img = cv2.imread(img_path)

    # generate the average number of BGR
    b_1d = np.full((480, 854, 1), 104)
    g_1d = np.full((480, 854, 1), 117)
    r_1d = np.full((480, 854, 1), 124)
    bgr_mean = np.concatenate((b_1d, g_1d), axis=2)
    bgr_mean = np.concatenate((bgr_mean, r_1d), axis=2)

    img = img - bgr_mean

    img_resized = transform.resize(img, [224, 224, 3])
    #state_t = vgg.extract_conv5(img_resized)

    state_t = get_state(current_path)


    # a tricky condition to check if current path is the last one
    while (len(os.listdir(current_path)) > 12):
      # generate the predicted action of current frame
      a_t = actor.choose_action(state_t)
      a_t_numpy = np.array([a_t]).reshape(1, 1)

      # import the correct negative action
      negative_action_path = os.path.join(current_path, "negative_action")
      f = open(negative_action_path, "r")
      negative_action = f.read()
      f.close()

      # generate the reward
      if abs(a_t - int(negative_action)) == 0:
        r_t = 8
      if abs(a_t - int(negative_action)) == 1:
        r_t = 4
      if abs(a_t - int(negative_action)) == 2:
        r_t = 2
      if abs(a_t - int(negative_action)) == 3:
        r_t = 0

      r_list = [r_t]
      r_list = np.array(r_list)
      r_list = r_list.reshape(1, 1)

      # turn to the next level
      current_path = os.path.join(current_path, "0")

      # generate the next state
      img_next_path = os.path.join(current_path, "img_t.jpg")
      img_next = cv2.imread(img_next_path)

      # generate the average number of BGR
      b_1d = np.full((480, 854, 1), 104)
      g_1d = np.full((480, 854, 1), 117)
      r_1d = np.full((480, 854, 1), 124)
      bgr_mean = np.concatenate((b_1d, g_1d), axis=2)
      bgr_mean = np.concatenate((bgr_mean, r_1d), axis=2)

      #img_next = img_next - bgr_mean

      img_next_resized = transform.resize(img_next, [224, 224, 3])
      #state_next = vgg.extract_conv5(img_next_resized)
      state_next = get_state(current_path)

      # update the critic model
      error = critic.learn(state_t, r_list, state_next)

      # update the actor model
      actor.learn(state_t, a_t_numpy, error)

      mean_loss = 0.9*mean_loss + 0.1*error

      print("loss:" + str(mean_loss) + "action:" + str(a_t) + ", reward:" + str(r_t))

      state_t = state_next

      count += 1


      pass

  if i%100 == 0:
    actor.save()
    critic.save()
    print("model saved")


