from Forwarding.OneshotForwarder import OneshotForwarder
from datasets.Util.Timer import Timer
from Measures import average_measures
from Log import log
import cv2
import numpy
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion
from parameter import *
import os
import copy
from skimage import transform

VOID_LABEL = 255

class OnlineAdaptingForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(OnlineAdaptingForwarder, self).__init__(engine)
    self.n_adaptation_steps = self.config.int("n_adaptation_steps", 12)
    self.adaptation_interval = self.config.int("adaptation_interval", 4)
    self.adaptation_learning_rate = self.config.float("adaptation_learning_rate")
    #self.posterior_positive_threshold = self.config.float("posterior_positive_threshold", 0.97)
    #self.distance_negative_threshold = self.config.float("distance_negative_threshold", 150.0)
    self.adaptation_loss_scale = self.config.float("adaptation_loss_scale", 0.1)
    self.debug = self.config.bool("adapt_debug", False)
    self.erosion_size = self.config.int("adaptation_erosion_size", 20)
    self.use_positives = self.config.bool("use_positives", True)
    self.use_negatives = self.config.bool("use_negatives", True)

  def _oneshot_forward_video(self, video_idx, save_logits):
    with Timer():

      # initialize part (original code)
      network = self.engine.test_network
      targets = network.raw_labels
      ys = network.y_softmax
      ys = self._adjust_results_to_targets(ys, targets)
      data = self.val_data
      # n_frames indicate the total number of all frames (including the first frame with ground)
      n_frames = data.num_examples_per_epoch()

      # get name of the current sequence and set the current path with this seq name
      path_repository = REPOSITORY_PATH
      name_seq = self._get_seq(video_idx)
      seq_path = os.path.join(path_repository, name_seq)

      current_path = os.path.join(path_repository, name_seq)

      # deal with the root directory of that video
      if os.path.isdir(current_path) == False:
        os.mkdir(current_path)
        f = open(os.path.join(current_path, "exist"), "w")
        f.close()

      # this directory is used to store the result mask of rach frame
      result_path = os.path.join(seq_path, "result")
      if os.path.isdir(result_path) == False:
        os.mkdir(result_path)

      # deal with the finetune network
      if os.path.exists(os.path.join(current_path, "finetune_model.ckpt.meta")):
        print("the finetune model exists, load this model now")
        self.engine.saver.restore(self.session, os.path.join(current_path, "finetune_model.ckpt"))
      else:
        print("no existing finetune model, finetune the network now")
        self._finetune(video_idx, n_finetune_steps=self.n_finetune_steps)
        # save the fintuned model
        self.engine.saver.save(self.session, os.path.join(current_path, "finetune_model.ckpt"))

      # temp mask of frame 0 (first frame with groundtruth), which is used for the negative pixels for the next frame update
      # last_mask

      iou_list = []
      negative_threshold_list = []
      positive_threshold_list = []

      n, measures, ys_argmax_val, logits_val, targets_val = self._process_forward_minibatch(data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=0)
      last_mask = targets_val[0]

      assert n == 1
      measures_video = []

      # img_t of frame 1, not the first frame (with groundtruth)!
      img_t = self._get_img(video_idx, 1)

      # get the distance information dt
      dt = self._get_dt(last_mask)
      dt = dt[:, :, 0]

      _, _, temp_ys_argmax_val, temp_logits_val, targets_val = self._process_forward_minibatch(data, network, save_logits, False, targets, ys, start_frame_idx=1)
      # generate the temp probability map of the current frame
      temp_map_t = temp_logits_val[0]
      # generate the label of current frame
      label_t = targets_val[0,:,:,0]
      # generate the result mask before update
      temp_mask_t = temp_ys_argmax_val[0,:,:,0]
      # generate the mask to show the different area of positive pixels
      positive_mask_t = self._get_positive_mask(temp_map_t, label_t)
      # generate the mask to show the different area of negative pixels
      negative_mask_t, _ = self._get_negative_mask(dt, temp_map_t, img_t)


      # generate the state of the current frame
      probability_map = temp_map_t
      probability_map = probability_map[:, :, 1]

      # generate the candidate adpatation areas with different threshold before cropping
      area_97 = probability_map > 0.97
      area_70 = probability_map > 0.7
      area_40 = probability_map > 0.4
      area_20 = probability_map > 0.2
      area_10 = probability_map > 0.1

      # generate the bounding box the area_10 (biggest candicate adaptation area)
      mask_bounding = numpy.full((480, 854), 1) * area_10
      mask_bounding = mask_bounding.astype(numpy.uint8)
      x, y, w, h = cv2.boundingRect(mask_bounding)

      # crop all candidate adaptation areas and the original image
      img = img_t
      img = img[y:y + h, x:x + w]
      area_97 = area_97[y:y + h, x:x + w]
      area_70 = area_70[y:y + h, x:x + w]
      area_40 = area_40[y:y + h, x:x + w]
      area_20 = area_20[y:y + h, x:x + w]
      area_10 = area_10[y:y + h, x:x + w]

      # mask (RGB image) of adaptation area with probability greater than 0.97
      area_97_d1 = area_97.reshape(h, w, 1)
      area_97_d2 = numpy.concatenate((area_97_d1, area_97_d1), axis=2)
      area_97_d3 = numpy.concatenate((area_97_d2, area_97_d1), axis=2)
      mask_97 = img * area_97_d3

      # mask (RGB image) of adaptation area with probability greater than 0.7
      area_70_d1 = area_70.reshape(h, w, 1)
      area_70_d2 = numpy.concatenate((area_70_d1, area_70_d1), axis=2)
      area_70_d3 = numpy.concatenate((area_70_d2, area_70_d1), axis=2)
      mask_70 = img * area_70_d3

      # mask (RGB image) of adaptation area with probability greater than 0.4
      area_40_d1 = area_40.reshape(h, w, 1)
      area_40_d2 = numpy.concatenate((area_40_d1, area_40_d1), axis=2)
      area_40_d3 = numpy.concatenate((area_40_d2, area_40_d1), axis=2)
      mask_40 = img * area_40_d3

      # mask (RGB image) of adaptation area with probability greater than 0.2
      area_20_d1 = area_20.reshape(h, w, 1)
      area_20_d2 = numpy.concatenate((area_20_d1, area_20_d1), axis=2)
      area_20_d3 = numpy.concatenate((area_20_d2, area_20_d1), axis=2)
      mask_20 = img * area_20_d3

      # mask (RGB image) of adaptation area with probability greater than 0.1
      area_10_d1 = area_10.reshape(h, w, 1)
      area_10_d2 = numpy.concatenate((area_10_d1, area_10_d1), axis=2)
      area_10_d3 = numpy.concatenate((area_10_d2, area_10_d1), axis=2)
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
      mask_97_feature = self.engine.vgg.extract_conv5(mask_97_resized)
      mask_70_feature = self.engine.vgg.extract_conv5(mask_70_resized)
      mask_40_feature = self.engine.vgg.extract_conv5(mask_40_resized)
      mask_20_feature = self.engine.vgg.extract_conv5(mask_20_resized)
      mask_10_feature = self.engine.vgg.extract_conv5(mask_10_resized)

      # reshape all feature maps
      mask_97_feature = numpy.reshape(mask_97_feature, (7, 7, 512))
      mask_70_feature = numpy.reshape(mask_70_feature, (7, 7, 512))
      mask_40_feature = numpy.reshape(mask_40_feature, (7, 7, 512))
      mask_20_feature = numpy.reshape(mask_20_feature, (7, 7, 512))
      mask_10_feature = numpy.reshape(mask_10_feature, (7, 7, 512))

      state_positive = numpy.concatenate((mask_97_feature, mask_70_feature), axis=2)
      state_positive = state_positive.reshape(1, 7, 7, 1024)

      state_negative = numpy.concatenate((mask_40_feature, mask_20_feature, mask_10_feature), axis=2)
      state_negative = state_negative.reshape(1, 7, 7, 1536)

      negative_state_t = state_negative
      positive_state_t = state_positive

      next_img = None
      next_temp_mask = None
      next_temp_map = None
      next_label = None
      next_positive_mask = None
      next_negative_mask = None

      ##################################################
      ### frame loop starts here #######################
      ##################################################

      # actions to records the action which has beed used
      actions = []

      for t in xrange(1, n_frames):
        # get the current action value
        a = self.engine.current_path.pop(0)
        actions.append(a)

        negative_a = self.engine.negative_actor.choose_action(negative_state_t)
        if negative_a == 0:
          n_t = 0.4
        elif negative_a == 1:
          n_t = 0.2
        elif negative_a == 2:
          n_t = 0.1
        elif negative_a == 3:
          n_t = 0.01

        positive_a = self.engine.positive_actor.choose_action(positive_state_t)
        if positive_a == 0:
          p_t = 0.97
        elif positive_a == 1:
          p_t = 0.7

        print("positive action: " + str(positive_a))
        print("negative action: " + str(negative_a))


        if positive_a==0 and negative_a ==0:
          negatives = self._adapt(video_idx, t, last_mask, temp_map_t, 150, 0.4, 0.97, 15, 5, t)
        elif positive_a==0 and negative_a ==1:
          negatives = self._adapt(video_idx, t, last_mask, temp_map_t, 150, 0.2, 0.97, 15, 5, t)
        elif positive_a==0 and negative_a ==2:
          negatives = self._adapt(video_idx, t, last_mask, temp_map_t, 150, 0.1, 0.97, 15, 5, t)
        elif positive_a==0 and negative_a ==3:
          negatives = self._adapt(video_idx, t, last_mask, temp_map_t, 150, 0.01, 0.7, 15, 5, t)
        elif positive_a==1 and negative_a ==0:
          negatives = self._adapt(video_idx, t, last_mask, temp_map_t, 150, 0.4, 0.7, 15, 5, t)
        elif positive_a==1 and negative_a ==1:
          negatives = self._adapt(video_idx, t, last_mask, temp_map_t, 150, 0.2, 0.7, 15, 5, t)
        elif positive_a==1 and negative_a ==2:
          negatives = self._adapt(video_idx, t, last_mask, temp_map_t, 150, 0.1, 0.7, 15, 5, t)
        elif positive_a==1 and negative_a ==3:
          negatives = self._adapt(video_idx, t, last_mask, temp_map_t, 150, 0.01, 0.7, 15, 5, t)

        # get the final mask of the current frame
        n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
        iou = measures[0].get("iou")

        iou_list.append(iou)

        # record related data (frame t) into repository
        current_path = self.record(t, current_path, img_t, label_t, temp_mask_t, last_mask, iou, a, temp_map_t, dt, positive_mask_t, negative_mask_t)

        # generate the "last_mask" of the next frame
        # assign the current mask (t) to replace the mask of the last frame (t-1)
        last_mask = ys_argmax_val[0]

        result_pixels = numpy.sum(last_mask)
        print result_pixels

        #last_mask = numpy.full((480, 854), 1) * (posteriors_val[0,:,:,1]>negative_threshold)
        last_mask = numpy.full((480, 854), 1) * (posteriors_val[0,:,:,1]> (negative_threshold*10) )
        last_mask = numpy.reshape(last_mask, (480, 854, 1))

        # save the result mask
        result_mask_path = os.path.join(result_path, str(t)+".png")
        cv2.imwrite(result_mask_path, last_mask*255)

        # prune negatives from last mask
        # negatives are None if we think that the target is lost
        if negatives is not None and self.use_negatives:
          last_mask[negatives] = 0

        # get the distance information dt
        dt = self._get_dt(last_mask)
        dt = dt[:, :, 0]

        # if the current frame is not the last frame in the sequence
        if t < n_frames-1:

          # generate the image of the next frame
          next_img = self._get_img(video_idx, t+1)

          _, next_meature, next_ys_argmax_val, next_logits_val, next_targets_val = self._process_forward_minibatch(data, network, save_logits, False, targets, ys, start_frame_idx=t + 1)

          # generate the temp probability map of the next frame
          next_temp_map = next_logits_val[0]

          # generate the label of current frame
          next_label = next_targets_val[0,:,:,0]

          # generate the result mask(before update) of the next frame
          next_temp_mask = next_ys_argmax_val[0,:,:,0]

          # generate the mask to show the different area of positive pixels
          next_positive_mask = self._get_positive_mask(next_temp_map, next_label)

          # generate the mask to show the different area of negative pixels
          next_negative_mask, _ = self._get_negative_mask(dt, next_temp_map, next_img, negative_threshold, t)


          # generate the state of the next frame

          probability_map = next_temp_map
          probability_map = probability_map[:, :, 1]

          # generate the candidate adpatation areas with different threshold before cropping
          area_97 = probability_map > 0.97
          area_70 = probability_map > 0.7
          area_40 = probability_map > 0.4
          area_20 = probability_map > 0.2
          area_10 = probability_map > 0.1

          # generate the bounding box the area_10 (biggest candicate adaptation area)
          mask_bounding = numpy.full((480, 854), 1) * area_10
          mask_bounding = mask_bounding.astype(numpy.uint8)
          x, y, w, h = cv2.boundingRect(mask_bounding)

          # crop all candidate adaptation areas and the original image
          img = next_img
          img = img[y:y + h, x:x + w]
          area_97 = area_97[y:y + h, x:x + w]
          area_70 = area_70[y:y + h, x:x + w]
          area_40 = area_40[y:y + h, x:x + w]
          area_20 = area_20[y:y + h, x:x + w]
          area_10 = area_10[y:y + h, x:x + w]

          # mask (RGB image) of adaptation area with probability greater than 0.97
          area_97_d1 = area_97.reshape(h, w, 1)
          area_97_d2 = numpy.concatenate((area_97_d1, area_97_d1), axis=2)
          area_97_d3 = numpy.concatenate((area_97_d2, area_97_d1), axis=2)
          mask_97 = img * area_97_d3

          # mask (RGB image) of adaptation area with probability greater than 0.7
          area_70_d1 = area_70.reshape(h, w, 1)
          area_70_d2 = numpy.concatenate((area_70_d1, area_70_d1), axis=2)
          area_70_d3 = numpy.concatenate((area_70_d2, area_70_d1), axis=2)
          mask_70 = img * area_70_d3

          # mask (RGB image) of adaptation area with probability greater than 0.4
          area_40_d1 = area_40.reshape(h, w, 1)
          area_40_d2 = numpy.concatenate((area_40_d1, area_40_d1), axis=2)
          area_40_d3 = numpy.concatenate((area_40_d2, area_40_d1), axis=2)
          mask_40 = img * area_40_d3

          # mask (RGB image) of adaptation area with probability greater than 0.2
          area_20_d1 = area_20.reshape(h, w, 1)
          area_20_d2 = numpy.concatenate((area_20_d1, area_20_d1), axis=2)
          area_20_d3 = numpy.concatenate((area_20_d2, area_20_d1), axis=2)
          mask_20 = img * area_20_d3

          # mask (RGB image) of adaptation area with probability greater than 0.1
          area_10_d1 = area_10.reshape(h, w, 1)
          area_10_d2 = numpy.concatenate((area_10_d1, area_10_d1), axis=2)
          area_10_d3 = numpy.concatenate((area_10_d2, area_10_d1), axis=2)
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
          mask_97_feature = self.engine.vgg.extract_conv5(mask_97_resized)
          mask_70_feature = self.engine.vgg.extract_conv5(mask_70_resized)
          mask_40_feature = self.engine.vgg.extract_conv5(mask_40_resized)
          mask_20_feature = self.engine.vgg.extract_conv5(mask_20_resized)
          mask_10_feature = self.engine.vgg.extract_conv5(mask_10_resized)

          # reshape all feature maps
          mask_97_feature = numpy.reshape(mask_97_feature, (7, 7, 512))
          mask_70_feature = numpy.reshape(mask_70_feature, (7, 7, 512))
          mask_40_feature = numpy.reshape(mask_40_feature, (7, 7, 512))
          mask_20_feature = numpy.reshape(mask_20_feature, (7, 7, 512))
          mask_10_feature = numpy.reshape(mask_10_feature, (7, 7, 512))

          state_positive = numpy.concatenate((mask_97_feature, mask_70_feature), axis=2)
          state_positive = state_positive.reshape(1, 7, 7, 1024)

          state_negative = numpy.concatenate((mask_40_feature, mask_20_feature, mask_10_feature), axis=2)
          state_negative = state_negative.reshape(1, 7, 7, 1536)

          negative_state_next = state_negative
          positive_state_next = state_positive

          img_t = next_img
          temp_mask_t = next_temp_mask
          temp_map_t = next_temp_map
          label_t = next_label
          positive_mask_t = next_positive_mask
          negative_mask_t = next_negative_mask

          negative_state_t = negative_state_next
          positive_state_t = positive_state_next

        assert n == 1
        assert len(measures) == 1
        measure = measures[0]
        print >> log.v5, "frame", t, ", IOU :", iou,  ", negative: ", negative_threshold, "positive: " , positive_threshold, "\n"

        measures_video.append(measure)

      ##################################################
      ### frame loop stops here #######################

      # information of the video

      #mean_iou = sum_iou / (TOTAL_FRAME -1)
      mean_iou = sum(iou_list) / (TOTAL_FRAME - 1)

      measures_video[:-1] = measures_video[:-1]
      measures_video = average_measures(measures_video)
      print >> log.v1, "sequence", video_idx + 1, data.video_tag(video_idx), measures_video

      # record the current path to existing path
      f = open(os.path.join(seq_path, "exist"), "a")
      f.write(str("%.5f"%mean_iou) + "    ")

      for a in actions:
        f.write(str(a) + " ")
      f.write("    ")
      for i_threshold in negative_threshold_list:
        f.write(str(i_threshold) + "  ")
      f.write("\n")
      f.close()

  def _adapt(self, video_idx, frame_idx, last_mask, current_posteriors, distance_threshold, negative_threshold, positive_threshold, total_step, steps, id):
    eroded_mask = grey_erosion(last_mask, size=(self.erosion_size, self.erosion_size, 1))
    dt = distance_transform_edt(numpy.logical_not(eroded_mask))

    adaptation_target = numpy.zeros_like(last_mask)
    adaptation_target[:] = VOID_LABEL

    positives = current_posteriors[:, :, 1] > positive_threshold
    if self.use_positives:
      adaptation_target[positives] = 1


    #threshold = self.distance_negative_threshold
    negatives_pre = dt > distance_threshold

    negatives = numpy.logical_or((dt[:,:,0] > distance_threshold), (current_posteriors[:,:,1] < negative_threshold))
    negatives = numpy.reshape(negatives, (480, 854, 1))
    if self.use_negatives:
      adaptation_target[negatives] = 0

    do_adaptation = eroded_mask.sum() > 0

    if self.debug:
      adaptation_target_visualization = adaptation_target.copy()
      adaptation_target_visualization[adaptation_target == 1] = 128
      if not do_adaptation:
        adaptation_target_visualization[:] = VOID_LABEL
      from scipy.misc import imsave
      folder = self.val_data.video_tag().replace("__", "/")
      imsave("forwarded/" + self.model + "/valid/" + folder + "/adaptation_%05d.png" % frame_idx,
             numpy.squeeze(adaptation_target_visualization))

    self.train_data.set_video_idx(video_idx)

    #for idx in xrange(self.n_adaptation_steps):
    #for idx in xrange(15):
    for idx in xrange(total_step):
      do_step = True
      #if idx % self.adaptation_interval == 0:
      if idx % steps == 0:
        if do_adaptation:
          feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
          feed_dict[self.train_data.get_label_placeholder()] = adaptation_target
          loss_scale = self.adaptation_loss_scale
          adaption_frame_idx = frame_idx
        else:
          print >> log.v4, "skipping current frame adaptation, since the target seems to be lost"
          do_step = False
      else:
        # mix in first frame to avoid drift
        # (do this even if we think the target is lost, since then this can help to find back the target)
        feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx=0, with_annotations=True)
        loss_scale = 1.0
        adaption_frame_idx = 0

      if do_step:
        loss, _, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale,
                                                  learning_rate=self.adaptation_learning_rate)
        assert n_imgs == 1
        print >> log.v4, "adapting on frame", adaption_frame_idx, "of sequence", video_idx + 1, \
            self.train_data.video_tag(video_idx), "loss:", loss
    if do_adaptation:
      return negatives
    else:
      return None

  def _get_seq(self, video_id):
    name_seqs = self.engine.name_seqs
    return name_seqs[video_id]

  def _get_img(self, video_id, img_id):
      imgs_path = self.engine.img_path_seqs
      img = cv2.imread(imgs_path[video_id][img_id], cv2.IMREAD_UNCHANGED)
      return img

  def _get_label(self, video_id, img_id):
    labels_path = self.engine.label_path_seqs
    label = cv2.imread(labels_path[video_id][img_id], cv2.IMREAD_UNCHANGED)
    return label

  def _get_dt(self, last_mask):
    eroded_mask = grey_erosion(last_mask, size=(self.erosion_size, self.erosion_size, 1))
    dt = distance_transform_edt(numpy.logical_not(eroded_mask))
    return dt

  def _get_positive_mask(self, temp_map_t, label_t):
    temp_map_t = temp_map_t[:,:,1]
    map = temp_map_t + label_t

    area_50_70_true = numpy.logical_and(map > 1.5, map < 1.7)

    area_70_97_true = numpy.logical_and(map > 1.7, map < 1.97)
    area_70_97_false = numpy.logical_and(map > 0.7, map < 0.97)

    area_97_100_true = map > 1.97

    final_mask = numpy.zeros((480, 854))

    final_mask += area_97_100_true * 30
    final_mask += area_70_97_true * 60
    final_mask += area_50_70_true * 150

    final_mask += area_70_97_false * 255

    return final_mask

  def _get_negative_mask(self, dt, temp_map_t, img_t, negative_threshold=0.1, t=1):
    full_1 = numpy.full((480, 854, 1), 1)

    negative_area1 = dt > 150

    negative_area2 = temp_map_t[:, :, 1]
    #negative_area2 = negative_area2 < 0.05
    negative_area2 = negative_area2 < negative_threshold

    negative_area_binary = numpy.logical_or(negative_area1, negative_area2)

    negative_area1 = numpy.reshape(negative_area1, (480, 854, 1))
    negative_area2 = numpy.reshape(negative_area2, (480, 854, 1))

    negative_2 = numpy.concatenate((full_1, full_1 * numpy.logical_not(negative_area2)), axis=2)
    negative_3 = numpy.concatenate((negative_2,full_1 * numpy.logical_not(negative_area1)), axis=2)
    #negative_3 = numpy.concatenate((negative_2, full_1), axis=2)

    negative_mask = img_t * negative_3
    return negative_mask, negative_area_binary

  def record(self, t, current_path, img_t, label_t, temp_mask_t, last_mask, iou, a, temp_map_t, dt, positive_mask, negative_mask):
      # save img_t
      path_img_t = os.path.join(current_path, "img_t.jpg")
      cv2.imwrite(path_img_t, img_t)

      # save label_t
      full = numpy.full((480, 854), 255)
      path_label_t = os.path.join(current_path, "label_t.png")
      cv2.imwrite(path_label_t, numpy.multiply(label_t, full))

      # save temp_mask_t
      full = numpy.full((480, 854), 255)
      path_temp_mask_t = os.path.join(current_path, "temp_mask_t.png")
      cv2.imwrite(path_temp_mask_t, numpy.multiply(temp_mask_t, full))

      # save last_mask
      path_last_mask = os.path.join(current_path, "last_mask.png")
      cv2.imwrite(path_last_mask, last_mask[:,:,0]*255)

      # save the maks to show the different area of positive pixels
      path_positive_mask = os.path.join(current_path, "positive_mask.png")
      cv2.imwrite(path_positive_mask, positive_mask)

      # save the maks to show the different area of negative pixels
      path_negative_mask = os.path.join(current_path, "negative_mask.jpg")
      cv2.imwrite(path_negative_mask, negative_mask)

      # save frame number
      path_info = os.path.join(current_path, "info")
      f = open(path_info, "w")
      f.write("frame: " + str(t) + "\n")
      f.write("path: " + current_path + "\n")
      f.close()

      # save final IOU after this action
      path_info = os.path.join(current_path, "iou_" + str(a))
      f = open(path_info, "w")
      f.write(str(iou))
      f.close()

      # save 2 channel probability map information
      post_path = os.path.join(current_path, "map.npy")
      numpy.save(post_path, temp_map_t)

      # save the distance information
      dt_path = os.path.join(current_path, "dt.npy")
      numpy.save(dt_path, dt)

      # save action as a directory named as the action number
      current_path = os.path.join(current_path, str(a))
      if os.path.isdir(current_path) == False:
          os.mkdir(current_path)

      return current_path


