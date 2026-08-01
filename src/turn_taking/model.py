"""Phase 3.3 turn-taking model: <10M params, <5ms single-frame inference.

Input per frame: up to 20 hashed-token ids (text, feature-hashing trick --
no pretrained tokenizer needed) + 4 prosody scalars (pitch, energy,
pause-so-far, speech rate) from src/features/pipeline.py's FrameFeatures.

`feature_mode` selects which half feeds the GRU -- this is Phase 3.4's
text-only/prosody-only/both ablation switch. One model class, one flag,
so the three ablation arms are guaranteed to differ only in their input,
never in incidental architecture drift between separately-written models.
"""

import torch
import torch.nn as nn

PAD_TOKEN_ID = 0
HASH_VOCAB_SIZE = 2000  # +1 slot reserved for PAD at index 0
MAX_TOKENS_PER_FRAME = 20
N_PROSODY_FEATURES = 4


def hash_token(token: str) -> int:
    """Deterministic feature-hashing index in [1, HASH_VOCAB_SIZE]."""
    return 1 + (hash(token) % HASH_VOCAB_SIZE)


class TurnTakingGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, embed_dim: int = 16, feature_mode: str = "both"):
        super().__init__()
        if feature_mode not in ("text", "prosody", "both"):
            raise ValueError(f"feature_mode must be text/prosody/both, got {feature_mode!r}")
        self.feature_mode = feature_mode
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(HASH_VOCAB_SIZE + 1, embed_dim, padding_idx=PAD_TOKEN_ID)

        input_dim = {
            "text": embed_dim,
            "prosody": N_PROSODY_FEATURES,
            "both": embed_dim + N_PROSODY_FEATURES,
        }[feature_mode]

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        """token_ids: [B,T,MAX_TOKENS_PER_FRAME] long. prosody: [B,T,4] float.
        Returns logits [B,T]; apply sigmoid for P(turn_complete)."""
        text_feat = None
        if self.feature_mode in ("text", "both"):
            embedded = self.token_embedding(token_ids)  # [B,T,20,E]
            mask = (token_ids != PAD_TOKEN_ID).float().unsqueeze(-1)  # [B,T,20,1]
            denom = mask.sum(dim=2).clamp(min=1.0)
            text_feat = (embedded * mask).sum(dim=2) / denom  # [B,T,E]

        if self.feature_mode == "text":
            x = text_feat
        elif self.feature_mode == "prosody":
            x = prosody
        else:
            x = torch.cat([text_feat, prosody], dim=-1)

        gru_out, _ = self.gru(x)  # [B,T,H]
        return self.head(gru_out).squeeze(-1)  # [B,T]

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
