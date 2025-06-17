import json
import os
import os.path as osp
import polars as pl
import torch

from collections import defaultdict
from torch_geometric.data import HeteroData, InMemoryDataset
from typing import Callable, List, Optional

class MySemIdDataset(InMemoryDataset):
    """
    A custom dataset loader for user interaction sequences already represented by semantic IDs.
    It expects two files in the `raw_dir`:
    1. `user_sem_ids.json`: A JSON file mapping user IDs to a list of their interactions, 
       where each interaction is a list of semantic IDs. e.g., {"user1": [[1,2,3], [4,5,6]]}
    2. `user_pids.json`: A JSON file mapping user IDs to a list of product IDs (pids),
       which must correspond to the semantic ID sequences. e.g., {"user1": ["pid_A", "pid_B"]}
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super(MySemIdDataset, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        # The names of your input data files
        return ['user_sem_ids.json', 'user_pids.json']

    @property
    def processed_file_names(self) -> str:
        # The name of the file that will be saved after processing
        return 'data.pt'

    def download(self) -> None:
        # No download needed for local files, so we pass.
        pass

    def _create_splits(self, user_sem_ids, user_pids, max_seq_len=20):
        """
        Splits the sequential data into train, eval, and test sets.
        - Train: All items except the last two.
        - Eval: Predicts the second-to-last item.
        - Test: Predicts the last item.
        """
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        
        # Create a mapping from user_str_id to an integer index
        all_users = list(user_sem_ids.keys())
        user_map = {user_id: i for i, user_id in enumerate(all_users)}

        for user_str_id, pids in user_pids.items():
            sem_id_sequences = user_sem_ids[user_str_id]
            user_id = user_map[user_str_id]

            if len(pids) < 3:  # Need at least 3 items to create train/eval/test splits
                continue

            # We use the product IDs for splitting and tracking original items
            # The semantic IDs will be carried alongside
            
            # --- Training Set ---
            train_pids = pids[:-2]
            train_sem_ids = sem_id_sequences[:-2]
            sequences["train"]["userId"].append(user_id)
            sequences["train"]["pids"].append(train_pids)
            sequences["train"]["sem_ids"].append(train_sem_ids)
            sequences["train"]["pids_fut"].append(pids[-2])
            sequences["train"]["sem_ids_fut"].append(sem_id_sequences[-2])

            # --- Evaluation Set ---
            eval_pids_context = pids[:-2] # Context is what's used for training
            eval_pids_target = pids[-2]   # Target is the next item
            eval_sem_ids_context = sem_id_sequences[:-2]
            eval_sem_ids_target = sem_id_sequences[-2]

            sequences["eval"]["userId"].append(user_id)
            sequences["eval"]["pids"].append(eval_pids_context[-max_seq_len:]) # Use last max_seq_len items
            sequences["eval"]["sem_ids"].append(eval_sem_ids_context[-max_seq_len:])
            sequences["eval"]["pids_fut"].append(eval_pids_target)
            sequences["eval"]["sem_ids_fut"].append(eval_sem_ids_target)

            # --- Test Set ---
            test_pids_context = pids[:-1] # Context includes the training and eval item
            test_pids_target = pids[-1]  # Target is the final item
            test_sem_ids_context = sem_id_sequences[:-1]
            test_sem_ids_target = sem_id_sequences[-1]
            
            sequences["test"]["userId"].append(user_id)
            sequences["test"]["pids"].append(test_pids_context[-max_seq_len:]) # Use last max_seq_len items
            sequences["test"]["sem_ids"].append(test_sem_ids_context[-max_seq_len:])
            sequences["test"]["pids_fut"].append(test_pids_target)
            sequences["test"]["sem_ids_fut"].append(test_sem_ids_target)

        for sp in splits:
            sequences[sp] = pl.from_dict(sequences[sp])

        return sequences


    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        # Load your raw data files
        with open(osp.join(self.raw_dir, self.raw_file_names[0]), 'r') as f:
            user_sem_ids = json.load(f)
        with open(osp.join(self.raw_dir, self.raw_file_names[1]), 'r') as f:
            user_pids = json.load(f)

        # Create train/eval/test splits
        sequences = self._create_splits(user_sem_ids, user_pids, max_seq_len=max_seq_len)
        
        # Store the sequences in the HeteroData object
        # We store both pids and the corresponding sem_ids
        data["user", "interacted_with", "item"].history = {
            k: {
                "userId": torch.tensor(v["userId"].to_numpy()),
                # We don't need to pad here, the SeqData loader will handle it
                "pids": v["pids"].to_list(),
                "sem_ids": v["sem_ids"].to_list(),
                # Future (target) items are single items, not sequences
                "pids_fut": torch.tensor(v.select(pl.col("pids_fut").flatten()).to_numpy().squeeze()),
                "sem_ids_fut": torch.tensor(v["sem_ids_fut"].to_list())
            }
            for k, v in sequences.items()
        }
        
        # Since we don't have item features (like text or brand), we create placeholders.
        # The model won't use them, but the data structure is expected.
        # We can build a simple map from unique pids to an integer index.
        all_pids = sorted(list(set(pid for pids_list in user_pids.values() for pid in pids_list)))
        pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}
        num_items = len(all_pids)
        
        data['item'].num_items = num_items
        # Placeholder for item features, as they won't be used by the VAE
        data['item'].x = torch.zeros(num_items, 1) 

        # Save the processed data object to a file
        self.save([data], self.processed_paths[0])
