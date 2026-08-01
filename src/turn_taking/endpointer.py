"""Phase 3.5: LearnedEndpointer -- integrates the trained TurnTakingGRU into
a streaming-callable component, for direct A/B comparison against the
fixed-threshold baseline (scripts/baseline_fixed_threshold_vad.py) on the
same eval scenarios.

Re-runs the model over the full growing frame sequence at each step() call
rather than threading GRU hidden state incrementally between calls. This
exactly reproduces what the model saw during training (one forward pass
over the whole sequence up to each supervised frame), which is what
genuinely-correct streaming inference must match -- and it's cheap here
since the model is ~90K params and utterances run tens of frames, not
thousands. True O(1) incremental inference (carrying hidden state instead
of re-running from t=0 each frame) is a Phase 4 production concern, not
needed for this A/B evaluation.

`fired` semantics mirror VADIterator's 'end' event and
scripts/baseline_fixed_threshold_vad.py's response-latency /
false-interruption-rate measurement: True exactly once, the first frame
P(turn_complete) crosses `threshold` since the last time speech resumed
(pause_so_far_ms reset to 0), so the two endpointers are measured the same
way.
"""

import torch

from features import FeaturePipeline
from .data import PAUSE_NORM_MS, PITCH_NORM_HZ, RATE_NORM_TPS
from .model import MAX_TOKENS_PER_FRAME, PAD_TOKEN_ID, hash_token


class LearnedEndpointer:
    def __init__(self, model, vad_model, asr_model, threshold: float = 0.6, device: str = "cpu", asr_refresh_every_n_frames: int = 5):
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = threshold
        self.pipeline = FeaturePipeline(vad_model, asr_model, asr_refresh_every_n_frames=asr_refresh_every_n_frames)
        self._token_ids_seq = []
        self._prosody_seq = []
        self._fired = False

    def reset(self) -> None:
        self._token_ids_seq = []
        self._prosody_seq = []
        self._fired = False

    def step(self, chunk) -> dict:
        feats = self.pipeline.step(chunk)

        ids = [hash_token(t) for t in feats.partial_tokens][:MAX_TOKENS_PER_FRAME]
        ids = ids + [PAD_TOKEN_ID] * (MAX_TOKENS_PER_FRAME - len(ids))
        self._token_ids_seq.append(ids)
        self._prosody_seq.append([
            feats.pitch_hz / PITCH_NORM_HZ,
            feats.energy_rms,
            feats.pause_so_far_ms / PAUSE_NORM_MS,
            min(feats.speech_rate_tps / RATE_NORM_TPS, 1.0),
        ])

        token_ids = torch.tensor([self._token_ids_seq], dtype=torch.long, device=self.device)
        prosody = torch.tensor([self._prosody_seq], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(token_ids, prosody)
            prob = torch.sigmoid(logits[0, -1]).item()

        if feats.pause_so_far_ms == 0:
            self._fired = False  # speech resumed: re-arm for the next pause

        # Only allowed to fire while VAD is actually observing silence.
        # Without this gate, a probability spike during active speech (the
        # partial-ASR-token flicker every asr_refresh_every_n_frames frames
        # can cause these) would count as a legitimate "turn complete"
        # decision -- confirmed as a real failure mode by direct evaluation:
        # some frames of active speech pushed P(turn_complete) into the
        # 0.6-0.69 range on the Phase 3.5 A/B eval set before this gate was added.
        in_pause = feats.pause_so_far_ms > 0
        just_fired = in_pause and prob >= self.threshold and not self._fired
        if just_fired:
            self._fired = True

        return {
            "frame_index": feats.frame_index,
            "t_ms": feats.t_ms,
            "prob": prob,
            "pause_so_far_ms": feats.pause_so_far_ms,
            "fired": just_fired,
        }
