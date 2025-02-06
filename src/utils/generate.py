from typing import List

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):
    MAX_SEQ_LENGTH = 1 << 30

    def __init__(self, stop_tokens: List[torch.Tensor]):
        super().__init__()
        self.stop_tokens = stop_tokens
        self.stopped = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        position = input_ids.shape[1]
        if self.stopped is None:
            self.stop_tokens = [
                tokens.to(input_ids.device) for tokens in self.stop_tokens
            ]
            self.stopped = (
                torch.ones((input_ids.shape[0],), dtype=torch.int32)
                * self.MAX_SEQ_LENGTH
            ).to(input_ids.device)

        for stop in self.stop_tokens:
            if input_ids.shape[1] < stop.shape[1]:
                continue
            matching = torch.all(stop == input_ids[:, -stop.shape[1] :], dim=1).to(
                dtype=torch.int32
            )
            self.stopped = torch.min(
                self.stopped, matching * position + (1 - matching) * self.MAX_SEQ_LENGTH
            )

        if torch.all(self.stopped != self.MAX_SEQ_LENGTH).item():
            return True

        return False


def build_stopping_criteria(stop_token_seqs: List[List[int]]):
    criteria = StoppingCriteriaSub(
        [
            torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)
            for tokens in stop_token_seqs
        ]
    )
    return StoppingCriteriaList([criteria])
