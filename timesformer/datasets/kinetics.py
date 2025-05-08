# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)
# logger.setLevel(logging.INFO)

# @DATASET_REGISTRY.register()
# class Kinetics(torch.utils.data.Dataset):
#     """
#     Kinetics video loader. Construct the Kinetics video loader, then sample
#     clips from the videos. For training and validation, a single clip is
#     randomly sampled from every video with random cropping, scaling, and
#     flipping. For testing, multiple clips are uniformaly sampled from every
#     video with uniform cropping. For uniform cropping, we take the left, center,
#     and right crop if the width is larger than height, or take top, center, and
#     bottom crop if the height is larger than the width.
#     """

#     def __init__(self, cfg, mode, num_retries=10):
#         """
#         Construct the Kinetics video loader with a given csv file. The format of
#         the csv file is:
#         ```
#         path_to_video_1 label_1
#         path_to_video_2 label_2
#         ...
#         path_to_video_N label_N
#         ```
#         Args:
#             cfg (CfgNode): configs.
#             mode (string): Options includes `train`, `val`, or `test` mode.
#                 For the train and val mode, the data loader will take data
#                 from the train or val set, and sample one clip per video.
#                 For the test mode, the data loader will take data from test set,
#                 and sample multiple clips per video.
#             num_retries (int): number of retries.
#         """
#         # Only support train, val, and test mode.
#         assert mode in [
#             "train",
#             "val",
#             "test",
#         ], "Split '{}' not supported for Kinetics".format(mode)
#         self.mode = mode
#         self.cfg = cfg

#         self._video_meta = {}
#         self._num_retries = num_retries
#         # For training or validation mode, one single clip is sampled from every
#         # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
#         # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
#         # the frames.
#         if self.mode in ["train", "val"]:
#             self._num_clips = 1
#         elif self.mode in ["test"]:
#             self._num_clips = (
#                 cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
#             )

#         logger.info("Constructing Kinetics {}...".format(mode))
#         self._construct_loader()

#     def _construct_loader(self):
#         """
#         Construct the video loader.
#         """
#         path_to_file = os.path.join(
#             self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
#         )
#         assert PathManager.exists(path_to_file), "{} dir not found".format(
#             path_to_file
#         )

#         self._path_to_videos = []
#         self._labels = []
#         self._spatial_temporal_idx = []
#         logger.info("Read Kinetics data from files ...")
#         with PathManager.open(path_to_file, "r") as f:
#             for clip_idx, path_label in enumerate(f.read().splitlines()):
#                 # print('path_label', path_label, path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR), self.cfg.DATA.PATH_LABEL_SEPARATOR)
#                 assert (
#                     len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
#                     == 2
#                 )
#                 path, label = path_label.split(
#                     self.cfg.DATA.PATH_LABEL_SEPARATOR
#                 )
#                 for idx in range(self._num_clips):
#                     self._path_to_videos.append(
#                         os.path.join(self.cfg.DATA.PATH_PREFIX, path)
#                     )
#                     self._labels.append(int(label))
#                     self._spatial_temporal_idx.append(idx)
#                     self._video_meta[clip_idx * self._num_clips + idx] = {}
#         assert (
#             len(self._path_to_videos) > 0
#         ), "Failed to load Kinetics split {} from {}".format(
#             self._split_idx, path_to_file
#         )
#         logger.info(
#             "Constructing kinetics dataloader (size: {}) from {}".format(
#                 len(self._path_to_videos), path_to_file
#             )
#         )

#     def __getitem__(self, index):
#         """
#         Given the video index, return the list of frames, label, and video
#         index if the video can be fetched and decoded successfully, otherwise
#         repeatly find a random video that can be decoded as a replacement.
#         Args:
#             index (int): the video index provided by the pytorch sampler.
#         Returns:
#             frames (tensor): the frames of sampled from the video. The dimension
#                 is `channel` x `num frames` x `height` x `width`.
#             label (int): the label of the current video.
#             index (int): if the video provided by pytorch sampler can be
#                 decoded, then return the index of the video. If not, return the
#                 index of the video replacement that can be decoded.
#         """
#         short_cycle_idx = None
#         # When short cycle is used, input index is a tupple.
#         if isinstance(index, tuple):
#             index, short_cycle_idx = index

#         if self.mode in ["train", "val"]:
#             # -1 indicates random sampling.
#             temporal_sample_index = -1
#             spatial_sample_index = -1
#             min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
#             max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
#             crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
#             if short_cycle_idx in [0, 1]:
#                 crop_size = int(
#                     round(
#                         self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
#                         * self.cfg.MULTIGRID.DEFAULT_S
#                     )
#                 )
#             if self.cfg.MULTIGRID.DEFAULT_S > 0:
#                 # Decreasing the scale is equivalent to using a larger "span"
#                 # in a sampling grid.
#                 min_scale = int(
#                     round(
#                         float(min_scale)
#                         * crop_size
#                         / self.cfg.MULTIGRID.DEFAULT_S
#                     )
#                 )
#         elif self.mode in ["test"]:
#             temporal_sample_index = (
#                 self._spatial_temporal_idx[index]
#                 // self.cfg.TEST.NUM_SPATIAL_CROPS
#             )
#             # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
#             # center, or right if width is larger than height, and top, middle,
#             # or bottom if height is larger than width.
#             spatial_sample_index = (
#                 (
#                     self._spatial_temporal_idx[index]
#                     % self.cfg.TEST.NUM_SPATIAL_CROPS
#                 )
#                 if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
#                 else 1
#             )
#             min_scale, max_scale, crop_size = (
#                 [self.cfg.DATA.TEST_CROP_SIZE] * 3
#                 if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
#                 else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
#                 + [self.cfg.DATA.TEST_CROP_SIZE]
#             )
#             # The testing is deterministic and no jitter should be performed.
#             # min_scale, max_scale, and crop_size are expect to be the same.
#             assert len({min_scale, max_scale}) == 1
#         else:
#             raise NotImplementedError(
#                 "Does not support {} mode".format(self.mode)
#             )
#         sampling_rate = utils.get_random_sampling_rate(
#             self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
#             self.cfg.DATA.SAMPLING_RATE,
#         )
#         # Try to decode and sample a clip from a video. If the video can not be
#         # decoded, repeatly find a random video replacement that can be decoded.
#         for i_try in range(self._num_retries):
#             video_container = None
#             try:
#                 video_container = container.get_video_container(
#                     self._path_to_videos[index],
#                     self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
#                     self.cfg.DATA.DECODING_BACKEND,
#                 )
#             except Exception as e:
#                 logger.info(
#                     "Failed to load video from {} with error {}".format(
#                         self._path_to_videos[index], e
#                     )
#                 )

#             # logger.info(video_container)
#             # Select a random video if the current video was not able to access.
#             if video_container is None:
#                 logger.warning(
#                     "Failed to meta load video idx {} from {}; trial {}".format(
#                         index, self._path_to_videos[index], i_try
#                     )
#                 )
#                 if self.mode not in ["test"] and i_try > self._num_retries // 2:
#                     # let's try another one
#                     index = random.randint(0, len(self._path_to_videos) - 1)
#                 else:
#                     index = 0
#                 continue

#             # Decode video. Meta info is used to perform selective decoding.
#             frames = decoder.decode(
#                 video_container,
#                 sampling_rate,
#                 self.cfg.DATA.NUM_FRAMES,
#                 temporal_sample_index,
#                 self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
#                 video_meta=self._video_meta[index],
#                 target_fps=self.cfg.DATA.TARGET_FPS,
#                 backend=self.cfg.DATA.DECODING_BACKEND,
#                 max_spatial_scale=min_scale,
#             )

#             # If decoding failed (wrong format, video is too short, and etc),
#             # select another video.
#             if frames is None:
#                 logger.warning(
#                     "Failed to decode video idx {} from {}; trial {}".format(
#                         index, self._path_to_videos[index], i_try
#                     )
#                 )
#                 if self.mode not in ["test"] and i_try > self._num_retries // 2:
#                     # let's try another one
#                     index = random.randint(0, len(self._path_to_videos) - 1)
#                 else:
#                     index = 0
#                 continue


#             label = self._labels[index]

#             # Perform color normalization.
#             frames = utils.tensor_normalize(
#                 frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
#             )

#             # T H W C -> C T H W.
#             frames = frames.permute(3, 0, 1, 2)
#             # Perform data augmentation.
#             frames = utils.spatial_sampling(
#                 frames,
#                 spatial_idx=spatial_sample_index,
#                 min_scale=min_scale,
#                 max_scale=max_scale,
#                 crop_size=crop_size,
#                 random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
#                 inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
#             )


#             if not self.cfg.MODEL.ARCH in ['vit']:
#                 frames = utils.pack_pathway_output(self.cfg, frames)
#             else:
#                 # Perform temporal sampling from the fast pathway.
#                 frames = torch.index_select(
#                      frames,
#                      1,
#                      torch.linspace(
#                          0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

#                      ).long(),
#                 )
#             save_path = self._path_to_videos[index].split('k400')[0] + 'k400_light' + self._path_to_videos[index].split('k400')[-1].split('.')[0] + '.pkl'
#             # print(self._path_to_videos[index].split('k400')[0] + 'K400_light' + self._path_to_videos[index].split('k400')[-1].split('.')[0] + '.pkl', label)
#             save_tensors_to_pickle(save_path, frames, label, index)
#             # break
#             return frames, label, index, {}
#         else:
#             raise RuntimeError(
#                 "Failed to fetch video after {} retries.".format(
#                     self._num_retries
#                 )
#             )

#     def __len__(self):
#         """
#         Returns:
#             (int): the number of videos in the dataset.
#         """
#         return len(self._path_to_videos)

@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        logger.info("Read Kinetics data from files ...")
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                # print('path_label', path_label, path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR), self.cfg.DATA.PATH_LABEL_SEPARATOR)
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            save_path = self._path_to_videos[index].split('k400')[0] + 'k400_light' + self._path_to_videos[index].split('k400')[-1].split('.')[0] + '.pkl'
            frames, label, _ = load_tensors_from_pickle(save_path)
            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                else:
                    index = 0
                continue

            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
    
import pickle
# import torch

def save_tensors_to_pickle(filename, frames, label, index):
    """
    Saves the given tensors (frames, label, index) into a pickle file.

    Args:
        filename (str): The name of the pickle file to save to.
        frames (torch.Tensor): The tensor representing the frames.
        label (torch.Tensor): The tensor representing the label.
        index (torch.Tensor): The tensor representing the index.
    """
    try:
        # Create a dictionary to hold the tensors
        data = {
            'frames': frames,
            'label': label,
            'index': index
        }

        # Open the file in binary write mode ('wb')
        with open(filename, 'wb') as f:
            # Use pickle.dump() to serialize and save the dictionary to the file
            pickle.dump(data, f)
        print(f"Successfully saved tensors to {filename}")

    except Exception as e:
        print(f"Error saving tensors to {filename}: {e}")
        # Consider re-raising the exception or returning an error indicator
        raise

def load_tensors_from_pickle(filename):
    """
    Loads tensors (frames, label, index) from a pickle file.

    Args:
        filename (str): The name of the pickle file to load from.

    Returns:
        tuple: A tuple containing the loaded tensors (frames, label, index), or None if an error occurs.
               Returns (None, None, None) on error.
    """
    try:
        # Open the file in binary read mode ('rb')
        with open(filename, 'rb') as f:
            # Use pickle.load() to deserialize and load the data from the file
            data = pickle.load(f)

        # Extract the tensors from the dictionary.  Handles the case where the keys might be missing.
        frames = data.get('frames')
        label = data.get('label')
        index = data.get('index')

        if frames is None or label is None or index is None:
             print(f"Error: Pickle file {filename} is missing one or more of 'frames', 'label', or 'index'.")
             return None, None, None

        return frames, label, index

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None, None, None
    except Exception as e:
        print(f"Error loading tensors from {filename}: {e}")
        return None, None, None  # Return None, None, None to indicate failure
    

def find_first_non_zero(list_):
  """
  Finds the first non-zero value in a list.

  Args:
    list_: The input list.

  Returns:
    The first non-zero value in the list, or None if no non-zero value is found.
  """
  for item in list_:
    if item != 0:
      return item
  return 0

@DATASET_REGISTRY.register()
class Rtmri75s(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=6, audio_mode=2):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self._audio_mode = audio_mode
        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
        # # INIT audio model
        # self._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        # self._audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        count = 0
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, video_audio in enumerate(f.read().splitlines()):
                assert (
                    len(video_audio.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = video_audio.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
                # if(count == 6):
                #     break
                # count += 1
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
                # logger.info(self._path_to_videos[index])
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            # # Load the pre-extract wav2vec2 embeddings
            # audio_embeddings = None
            # try:
            #     input_audio, sample_rate = librosa.load(self._path_to_audios[index],  sr=16000)
            #     aud_start_idx = 0
            #     if(self._audio_mode == 1):
            #         audio_embeddings = torch.load(self._path_to_audios[index])[0]
            #         audio_embeddings = audio_embeddings[aud_start_idx : aud_start_idx + self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE]
            #         assert audio_embeddings.shape[0] == self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE, "Not the right audio embedding shape, retrying..."
            #         audio_label = torch.mean(audio_embeddings, dim=0) 

            #     elif(self._audio_mode == 2):
            #         input_values = self._processor(input_audio, return_tensors="pt", sampling_rate=sample_rate).input_values
            #         # retrieve logits
            #         with torch.no_grad():
            #             logits = self._audio_model(input_values).logits
            #         predicted_ids = torch.argmax(logits, dim=-1).detach()
            #         ## there might be multiple phoneme exist in video period, we dont want to make it multiclass problems, so we choose first non-zero class label as main class for the entire period 
            #         audio_label = find_first_non_zero(predicted_ids[0][aud_start_idx : aud_start_idx + self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE])
            #     audio_label = 3
            # except Exception as e:
            #     logger.warning(
            #         "Failed to meta load audio idx {} from {}; trial {}".format(
            #             index, self._path_to_audios[index], i_try
            #         )
            #     )
            #     if self.mode not in ["test"] and i_try > self._num_retries // 2:
            #         # let's try another one
            #         index = random.randint(0, len(self._path_to_videos) - 1)
            #     continue

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue


            label = self._labels[index]

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )


            if not self.cfg.MODEL.ARCH in ['vit']:
                frames = utils.pack_pathway_output(self.cfg, frames)
            else:
                # Perform temporal sampling from the fast pathway.
                frames = torch.index_select(
                     frames,
                     1,
                     torch.linspace(
                         0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                     ).long(),
                )
            # save_path = self._path_to_videos[index].split('k400')[0] + 'k400_light' + self._path_to_videos[index].split('k400')[-1].split('.')[0] + '.pkl'
            # # print(self._path_to_videos[index].split('k400')[0] + 'K400_light' + self._path_to_videos[index].split('k400')[-1].split('.')[0] + '.pkl', label)
            # save_tensors_to_pickle(save_path, frames, label, index)
            # break
            print(self._path_to_videos[index])
            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)