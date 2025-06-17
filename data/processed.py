import gin
import os
import random
import torch

from data.amazon import AmazonReviews
from data.ml1m import RawMovieLens1M
from data.ml32m import RawMovieLens32M
from data.schemas import SeqBatch
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"


@gin.constants_from_enum
class RecDataset(Enum):
    AMAZON = 1
    ML_1M = 2
    ML_32M = 3


DATASET_NAME_TO_RAW_DATASET = {
    RecDataset.AMAZON: AmazonReviews,
    RecDataset.ML_1M: RawMovieLens1M,
    RecDataset.ML_32M: RawMovieLens32M
}


DATASET_NAME_TO_MAX_SEQ_LEN = {
    RecDataset.AMAZON: 20,
    RecDataset.ML_1M: 200,
    RecDataset.ML_32M: 200
}


class ItemData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        train_test_split: str = "all",
        **kwargs
    ) -> None:
        
        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, **kwargs)
        
        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)
        
        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(raw_data.data["item"]["x"][:,0], dtype=bool)

        self.item_data, self.item_text = raw_data.data["item"]["x"][filt], raw_data.data["item"]["text"][filt]

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=bool)
        )


class SeqData(Dataset):
    def __init__(
        self,
        root: str,
        sem_id_dim: int,
        *args,
        is_train: bool = True,
        subsample: bool = False,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        **kwargs
    ) -> None:
        
        assert (not subsample) or is_train, "Can only subsample on training split."

        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, **kwargs)

        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)

        split = "train" if is_train else "test"
        self.subsample = subsample
        self.sequence_data = raw_data.data[("user", "rated", "item")]["history"][split]

        if not self.subsample:
            self.sequence_data["itemId"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(l[-max_seq_len:]) for l in self.sequence_data["itemId"]],
                batch_first=True,
                padding_value=-1
            )

        self._max_seq_len = max_seq_len
        # self.item_data = raw_data.data["item"]["x"]
        self.split = split

        self.sem_id_dim = sem_id_dim
        # self.item_data now stores the dimension of a semantic ID
        self.item_data = self.sem_id_dim # This replaces the old self.item_data line
        
        # Also, modify the __init__ method in SeqData to store the semantic ID dimension
        # Find this line: self.item_data = raw_data.data["item"]["x"]
        # and add the following right after it:
        # self.item_data = model.sem_id_dim # Or get it from config
    
    
    @property
    def max_seq_len(self):
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
  
    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]
        
        # --- START OF MODIFICATIONS ---

        # We now fetch the pre-computed semantic IDs instead of item IDs
        # The 'subsample' logic for training is more complex with pre-tokenized data,
        # so we will disable it for simplicity and rely on the fixed-length sequences.
        # If subsampling is critical, the logic in `my_dataset.py` would need to be enhanced.
    
        # Pad a sequence of semantic ID lists to a fixed total length
        seq_sem_ids = self.sequence_data["sem_ids"][idx]
    
        # Flatten the list of lists and pad
        flat_sem_ids = [sid for sublist in seq_sem_ids for sid in sublist]
        padded_len = self.max_seq_len * self.item_data # item_data now holds sem_id_dim
    
        # Pad sequence to max length
        padded_flat_sem_ids = flat_sem_ids + [-1] * (padded_len - len(flat_sem_ids))
        item_sem_ids = torch.tensor(padded_flat_sem_ids)
    
        # Get the semantic IDs for the future/target item
        item_sem_ids_fut = self.sequence_data["sem_ids_fut"][idx]

        # Create the sequence mask based on the original (un-flattened) sequence length
        seq_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        seq_mask[:len(seq_sem_ids)] = True

        # The `x` fields of SeqBatch will now carry the semantic IDs.
        return SeqBatch(
            user_ids=user_ids,
            ids=-1 * torch.ones_like(user_ids),  # Original item IDs not strictly needed by model
            ids_fut=-1 * torch.ones_like(user_ids),
            x=item_sem_ids,
            x_fut=item_sem_ids_fut,
            seq_mask=seq_mask
        )


if __name__ == "__main__":
    dataset = ItemData("dataset/amazon", dataset=RecDataset.AMAZON, split="beauty", force_process=True)
    dataset[0]
    import pdb; pdb.set_trace()
