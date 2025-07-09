"""
Custom Dataset

@author: ktsumura
"""
from typing import List

import torch
from torch.utils.data import Dataset

from training.dataset.transforms import ComposeAPI
from training.neo_dataset.neo_data_point import NeoDataPoint
from training.neo_dataset.readdb.db_reader_state import DbReaderState
from training.neo_dataset.readdb.hdf5_multi_reader import HDF5MultiReader
from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData


class NeoDataset(Dataset):
    def __init__(self,
                 db_path_list: List[str],
                 transforms: ComposeAPI,
                 num_gpus: int=1,
                 batch_size: int=1,
                 epoch_size: int=1,
                 image_size: int=1024,
                 shuffle:bool=True):
        self._db_reader_state = DbReaderState(db_path_list, num_gpus, batch_size, epoch_size, shuffle)
        self._db_reader = HDF5MultiReader(self._db_reader_state, transforms, image_size)

        # Calculate the DB size
        steps_per_epoch = self._db_reader_state.calc_steps_per_epoch()
        self._db_size = steps_per_epoch * batch_size

    def __len__(self):
        return self._db_size

    def __getitem__(self, idx):
        return self._db_reader[idx]

    def set_epoch(self, epoch_idx):
        self._db_reader_state.init_epoch(epoch_idx)

    @staticmethod
    def collate_fn(
            batch: List[NeoDataPoint],
            dict_key,
    ) -> BatchedVideoDatapoint:
        """
        The method is equivalent of training.utils.data_utils.collate_fn

        Args:
            batch: A list of VideoDatapoint instances.
            dict_key (str): A string key used to identify the batch.
        """
        # [Channel, Depth(=Frame), Height, Width] x Batch Size
        img_batch = [video.image for video in batch]
        mask_batch = [video.label for video in batch]

        # [Frame, Batch, Channel(=3), Height, Width]
        img_batch = torch.stack(img_batch, dim=0).permute((2, 0, 1, 3, 4))
        img_batch = img_batch.repeat(1, 1, 3, 1, 1)

        # [Frame, Batch * Object(=1), Height, Width]
        mask_batch = torch.stack(mask_batch, dim=0).permute((2, 0, 1, 3, 4))
        mask_batch = torch.squeeze(mask_batch, axis=2)

        # The following error occurs down the road if img_batch and mask_batch are not moved to CPU.
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        img_batch = img_batch.detach().to('cpu')
        mask_batch = mask_batch.detach().to('cpu')

        num_frames = img_batch.shape[0]
        # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
        step_t_objects_identifier = [[] for _ in range(num_frames)]
        step_t_frame_orig_size = [[] for _ in range(num_frames)]

        step_t_masks = [[] for _ in range(num_frames)]
        step_t_obj_to_frame_idx = [
            [] for _ in range(num_frames)
        ]  # List to store frame indices for each time step

        for video_idx, video in enumerate(batch):
            orig_video_id = video.neo_data_info.data_id
            orig_frame_size = video.neo_data_info.size

            for frame_idx in range(0, num_frames):
                orig_obj_id = video.neo_data_info.sam2_contour_type.to_integer()
                orig_frame_idx = frame_idx
                step_t_obj_to_frame_idx[frame_idx].append(
                    torch.tensor([frame_idx, video_idx], dtype=torch.int)
                )
                #step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[frame_idx].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[frame_idx].append(torch.tensor(orig_frame_size))

        obj_to_frame_idx = torch.stack(
            [
                torch.stack(obj_to_frame_idx, dim=0)
                for obj_to_frame_idx in step_t_obj_to_frame_idx
            ],
            dim=0,
        )
        #masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
        objects_identifier = torch.stack(
            [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
        )
        frame_orig_size = torch.stack(
            [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
        )

        return BatchedVideoDatapoint(
            img_batch=img_batch,
            obj_to_frame_idx=obj_to_frame_idx,
            masks=mask_batch,
            metadata=BatchedVideoMetaData(
                unique_objects_identifier=objects_identifier,
                frame_orig_size=frame_orig_size,
            ),
            dict_key=dict_key,
            batch_size=[num_frames],
        )