from jaxtyping import Float, Int64
from torch import Tensor

TokenIds = Int64[Tensor, "seq_len"]
BatchTokenIds = Int64[Tensor, "batch_size seq_len"]
TokenProbs = Float[Tensor, "seq_len vocab_size"]
BatchTokenProbs = Float[Tensor, "batch_size seq_len vocab_size"]
PrefixCache = tuple[tuple[Float[Tensor, "batch_size *"]]]
