import Measures
from Forwarding.Forwarder import ImageForwarder
import time
from math import ceil
from Log import log
from parameter import *
import copy


class OneshotForwarder(ImageForwarder):
  def __init__(self, engine):
    super(OneshotForwarder, self).__init__(engine)
    self.val_data = self.engine.valid_data
    if hasattr(self.engine, "train_data"):
      self.train_data = self.engine.train_data
    self.trainer = self.engine.trainer
    self.forward_interval = self.config.int("forward_interval", 9999999)
    self.forward_initial = self.config.bool("forward_initial", False)
    self.n_finetune_steps = self.config.int("n_finetune_steps", 40)
    self.video_range = self.config.int_list("video_range", [])
    self.video_ids = self.config.int_list("video_ids", [])
    assert len(self.video_range) == 0 or len(self.video_ids) == 0, "cannot specify both"
    self.save_oneshot = self.config.bool("save_oneshot", False)
    self.lucid_interval = self.config.int("lucid_interval", -1)
    self.lucid_loss_scale = self.config.float("lucid_loss_scale", 1.0)

  def _maybe_adjust_output_layer_for_multiple_objects(self):
    if not self.config.bool("adjustable_output_layer", False):
      return
    assert hasattr(self.val_data, "get_number_of_objects_for_video")
    n_objects = self.val_data.get_number_of_objects_for_video()
    print >> log.v3, "adjusting output layer for", n_objects, "objects"
    self.engine.train_network.get_output_layer().adjust_weights_for_multiple_objects(self.session, n_objects)








  def forward(self, network, data, save_results=True, save_logits=False):
    if len(self.video_range) != 0:
      video_ids = range(self.video_range[0], self.video_range[1])
    elif len(self.video_ids) != 0:
      video_ids = self.video_ids
    else:
      video_ids = range(0, self.train_data.n_videos())

    for video_idx in video_ids:
      tag = self.train_data.video_tag(video_idx)
      print >> log.v4, "finetuning on", tag

      # reset weights and optimizer for next video
      self.engine.try_load_weights()
      self.engine.reset_optimizer()
      self.val_data.set_video_idx(video_idx)
      self._maybe_adjust_output_layer_for_multiple_objects()

      e = TOTAL_FRAME - 2

      '''
      # generate all action paths
      current_list = [[0], [1]]
      #current_list = [[0], [1], [2]]
      final_list =[]
      for i in range(e):
        final_list = []
        for path in current_list:
          path_0 = copy.deepcopy(path)
          path_1 = copy.deepcopy(path)
          #path_2 = copy.deepcopy(path)
          path_0.append(0)
          path_1.append(1)
          #path_2.append(2)
          final_list.append(path_0)
          final_list.append(path_1)
          #final_list.append(path_2)
        current_list = copy.deepcopy(final_list)
      action_path_list = final_list

      action_path_list.sort(reverse=True)
      '''

      # generate a fixed action path, the number of actions euqals to TOTAL_FRAME - 1
      #action_path_list = [[0,0,0,0,0,0]]

      action_path_list =[[]]
      for i in range(TOTAL_FRAME-1):
        action_path_list[0].append(0)

      self.engine.current_path = []

      for path in action_path_list:
        self.engine.current_path = copy.deepcopy(path)
        self._oneshot_forward_video(video_idx, save_logits)
        self.engine.try_load_weights()
        self.engine.reset_optimizer()















  def _oneshot_forward_video(self, video_idx, save_logits):
    forward_interval = self.forward_interval
    if forward_interval > self.n_finetune_steps:
      forward_interval = self.n_finetune_steps
      n_partitions = 1
    else:
      n_partitions = int(ceil(float(self.n_finetune_steps) / forward_interval))
    if self.forward_initial:
      self._base_forward(self.engine.test_network, self.val_data, save_results=False, save_logits=False)
    for i in xrange(n_partitions):
      start = time.time()
      self._finetune(video_idx, n_finetune_steps=min(forward_interval, self.n_finetune_steps),
                     start_step=forward_interval * i)
      save_results = self.save_oneshot and i == n_partitions - 1
      save_logits_here = save_logits and i == n_partitions - 1
      self._base_forward(self.engine.test_network, self.val_data, save_results=save_results,
                         save_logits=save_logits_here)
      end = time.time()
      elapsed = end - start
      print >> log.v4, "steps:", forward_interval * (i + 1), "elapsed", elapsed

  def _base_forward(self, network, data, save_results, save_logits):
    super(OneshotForwarder, self).forward(self.engine.test_network, self.val_data, save_results, save_logits)

  def _finetune(self, video_idx, n_finetune_steps, start_step=0):
    frame_idx = 0
    tag = self.train_data.video_tag(video_idx)
    self.train_data.set_video_idx(video_idx)
    for idx in xrange(start_step, start_step + n_finetune_steps):
      if self.lucid_interval != -1 and idx % self.lucid_interval == 0:
        print >> log.v5, "lucid example"
        feed_dict = self.train_data.get_lucid_feed_dict()
        loss_scale = self.lucid_loss_scale
      else:
        feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
        loss_scale = 1.0
      loss, measures, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale)
      loss /= n_imgs
      iou = Measures.calc_iou(measures, n_imgs, [0])
      print >> log.v5, "finetune on", tag, idx, "/", start_step + n_finetune_steps, "loss:", loss, " iou:", iou
