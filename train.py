import os
import copy
import numpy
from skimage import transform
from parameter import *
from actor import *
from critic import *
from new_vgg19 import *

def get_mean_list(l):
  s = 0
  for i in l:
    s += i
  return s/len(l)

def _get_reward(iou):
  #r = iou * iou * 5000 
  if iou > 0.1
    r = iou + 1
  else:
    r = -1
  return r

# the times of update
count = 0

# initialize the data (path)

dataset_path = DATSET_PATH

# name_seqs: the list of all sequence names
name_seqs = []
with open(NAME_SEQ, 'r') as f:
  name_seqs = f.read().splitlines()
num_seqs = len(name_seqs)

# path_seqs: the list consists of several other lists(sequence), each internal is an another list consists of paths of each frame in this sequence
# noth that it contains all frame in the dataset, not only 10 frames!
path_seqs = []
for name_seq in name_seqs:
  path_seq = os.path.join(dataset_path, name_seq)
  name_imgs = os.listdir(path_seq)
  name_imgs.sort()

  path_imgs = []
  for name_img in name_imgs:
    path_img = os.path.join(path_seq, name_img)
    path_imgs.append(path_img)
  path_seqs.append(path_imgs)


# initialize all models
vgg = VGG19()
vgg.restore()

actor = Actor(2, LR_A)
critic = Critic(LR_C)

if FIRST_TRAIN == True:
    actor.initialze()
    actor.save()
    critic.initialze()
    critic.save()

actor.restore()
critic.restore()

# main loop
# 1st loop: epoch
# 2nd loop: sequence
# 3rd loop: frame

path_repository = REPOSITORY_PATH

total_accuray = 0 # overall average iou of all videos

for i in range(ITER_EPOCH):

  t_acuracy = 0 # average iou of all videos in this training epoch

  for j in range(num_seqs):

    # these lists are used to store data of each frame
    state_t_list = []
    new_state_t_list = []
    a_list = []
    iou_list = []
    r_list = []

    current_seq_name = name_seqs[j] # name of the current sequence
    current_path = os.path.join(path_repository, current_seq_name) # path of current sequence in repository
    t_current_path = copy.deepcopy(current_path) # a copy

    # process for 9 times (9 frame except the first frame with the groundtruth)
    for k in range(1, 10):
      current_img_path = path_seqs[j][k] # path of the current image of the current sequence

      # get the state of current frame
      img = cv2.imread(os.path.join(t_current_path, "img_t.jpg"), cv2.IMREAD_UNCHANGED)
      img_height, img_width = img.shape[:2]
      img = numpy.array(img)

      mask = cv2.imread(os.path.join(t_current_path, "temp_mask_t.png"), cv2.IMREAD_UNCHANGED)
      mask = mask / 255

      augment_rate = AUGMENT_RATE
      x, y, w, h = cv2.boundingRect(mask)
      y_top = (y - augment_rate * h) if ((y - augment_rate * h) > 0) else 0
      y_bottom = (y + (1 + augment_rate) * h) if ((y + (1 + augment_rate) * h) < img_height) else img_height
      x_left = (x - augment_rate * w) if ((x - augment_rate * w) > 0) else 0
      x_right = (x + (1 + augment_rate) * w) if ((x + (1 + augment_rate) * w) < img_width) else img_width

      mask_reshaped = mask.reshape(img_height, img_width, 1)
      mask_2c = numpy.concatenate((mask_reshaped, mask_reshaped), axis=2)
      mask_3c = numpy.concatenate((mask_2c, mask_reshaped), axis=2)

      # the whole frame where only mask area is visible (3 channels)
      frame_mask = numpy.multiply(img, mask_3c)

      # the patch where only mask area is visble (3 channels)
      state_mask = frame_mask[int(y_top):int(y_bottom), int(x_left):int(x_right)]

      # the path where each pixel is visible (3 channels)
      state_patch = img[int(y_top):int(y_bottom), int(x_left):int(x_right)]

      state_patch_reshaped = transform.resize(state_patch, [224, 224, 3])
      state_mask_reshaped = transform.resize(state_mask, [224, 224, 3])

      patch_feature = vgg.extract_conv5(state_patch_reshaped)
      patch_feature = numpy.reshape(patch_feature, (7, 7, 512))

      mask_feature = vgg.extract_conv5(state_mask_reshaped)
      mask_feature = numpy.reshape(mask_feature, (7, 7, 512))

      new_state = numpy.concatenate((patch_feature, mask_feature), axis=2)
      new_state_t_list.append(new_state)

      current_a = actor.choose_action(numpy.reshape(new_state, (1, 7, 7, 1024)))
      a_list.append(current_a)

      if os.path.isdir(os.path.join(t_current_path, str(current_a))) != False:

        # add iou
        f = open(os.path.join(t_current_path, "iou_" + str(current_a)), "r")
        exist_iou = float(f.read())
        f.close()
        iou_list.append(exist_iou)
        t_current_path = os.path.join(t_current_path, str(current_a))
      else:
        break

    if len(iou_list) == 9:
      for i in range(len(new_state_t_list) - 1):
        current_iou = iou_list[i]
        r = _get_reward(current_iou)
        r_list.append(r)
        if SHOW_ACTION == True:
          print a_list[i],
      if SHOW_ACTION == True:
        print a_list[-1]

      r_np = numpy.array(r_list)
      r_np = numpy.reshape(r_np, (BTACH_SIZE - 1, 1))
      a_np = numpy.array(a_list[0:-1])
      a_np = numpy.reshape(a_np, (BTACH_SIZE - 1, 1))

      state_t_np = numpy.array(state_t_list[0:-1])
      next_state_np = numpy.array(state_t_list[1:])

      new_state_t_np = numpy.array(new_state_t_list[0:-1])
      new_next_state_np = numpy.array(new_state_t_list[1:])

      td_error = critic.learn(new_state_t_np, r_np, new_next_state_np)
      actor.learn(new_state_t_np, a_np, td_error)

      average_iou = get_mean_list(iou_list)
      t_acuracy += average_iou

  # save RL model
  if (count % SAVE_INTERVAL == 0) and (count > 0):
    actor.save()
    critic.save()
    print("RL model saved")

  # change the learning rate
  if (count % LR_INTERVAL == 0) and (count > 0):
    actor.lr *= LR_RATE
    critic.lr *= LR_RATE

  count += 1

  total_accuray = 0.95 * total_accuray + 0.05 * (t_acuracy / num_seqs)
  print("\n" + "epoch: " + str(count))
  print("overall: " + str(total_accuray))
  print("current: " + str(t_acuracy / num_seqs))








