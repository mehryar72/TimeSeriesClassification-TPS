from dataclasses import dataclass


@dataclass
class TUPEConfig:
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 128
    d_head: int = 0
    max_len: int = 256
    dropout: float = 0.1
    expansion_factor: int = 1
    relative_bias: bool = True
    bidirectional_bias: bool = True
    num_buckets: int = 32
    max_distance: int = 128

    def __post_init__(self):
        d_head, remainder = divmod(self.d_model, self.num_heads)
        assert remainder == 0, "`d_model` should be divisible by `num_heads`"
        self.d_head = d_head
