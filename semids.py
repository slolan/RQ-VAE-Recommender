import torch

from data.schemas import SeqBatch, TokenizedSeqBatch
from torch import nn, Tensor

class PassThroughTokenizer(nn.Module):
    """
    A tokenizer that does not use a VAE. It assumes the input batch already
    contains the semantic IDs and simply formats them into the required
    TokenizedSeqBatch structure for the decoder model.
    """
    def __init__(
        self,
        sem_id_dim: int,
        codebook_size: int,
        # The following parameters are for API compatibility with the training script
        # but are not used by this tokenizer.
        input_dim: int = 0,
        output_dim: int = 0,
        hidden_dims: list = None,
        n_layers: int = 0,
        n_cat_feats: int = 0,
        commitment_weight: float = 0.0,
        rqvae_weights_path: str = None,
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False
    ) -> None:
        super().__init__()
        # These are the only two parameters we actually need from the config
        self._sem_ids_dim = sem_id_dim
        self.codebook_size = codebook_size
        print("Initialized PassThroughTokenizer: RQ-VAE is DISABLED.")

    @property
    def sem_ids_dim(self):
        return self._sem_ids_dim

    def precompute_corpus_ids(self, item_dataset) -> None:
        # Not needed since we are not encoding a corpus. We do nothing.
        print("Skipping precomputation of corpus IDs.")
        pass
    
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        # This function is used for verified generation. For now, we can
        # default to returning True, or implement a check against your known item IDs.
        # Returning True for all is simpler and allows generation to proceed.
        return torch.ones(*sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device)

    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        """
        Takes a batch containing pre-computed semantic IDs and formats it.
        The `batch.x` field is expected to hold the semantic IDs for the sequence,
        and `batch.x_fut` holds the semantic IDs for the target item.
        """
        sem_ids = batch.x.long()  # Input semantic IDs for the historical sequence
        sem_ids_fut = batch.x_fut.long() # Input semantic IDs for the future item

        B, N_plus_D = sem_ids.shape
        _B_fut, D = sem_ids_fut.shape

        # The sequence mask should correspond to the original items, not the flattened sem_ids
        # We expand it to cover all semantic IDs for each valid item.
        seq_mask = batch.seq_mask.repeat_interleave(self.sem_ids_dim, dim=1)

        token_type_ids = torch.arange(self.sem_ids_dim, device=sem_ids.device).repeat(B, N_plus_D // self.sem_ids_dim)
        token_type_ids_fut = torch.arange(self.sem_ids_dim, device=sem_ids.device).repeat(B, 1)

        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )
