from typing import List
from transformers import PreTrainedTokenizer, LogitsWarper, StoppingCriteria
import torch

"""
This was forked and developed upon 
https://github.com/1rgs/jsonformer
All credit for the great idea and implementation goes to the original author
"""

class StringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _,
    ) -> bool:
        if len(input_ids[0]) <= self.prompt_length:
            return False

        last_token_id = input_ids[0][-1]
        last_token = self.tokenizer.decode(last_token_id, skip_special_tokens=True)

        result = '"' in last_token

        return result


class NumberStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        precision: int = 3,
    ):
        self.tokenizer = tokenizer
        self.precision = precision
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        decoded = self.tokenizer.decode(
            input_ids[0][self.prompt_length :], skip_special_tokens=True
        )

        if decoded.count(".") > 1:
            return True

        if (
            decoded.count(".") == 1
            and len(decoded.strip().split(".")[1]) > self.precision
        ):
            return True

        if (
            len(decoded) > 1
            and any(c.isdigit() for c in decoded)
            and decoded[-1] in [" ", "\n"]
        ):
            return True

        return False

class OutputNumbersTokens(LogitsWarper):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt: str):
        self.tokenizer = tokenizer
        self.tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)

        for _, token_id in tokenizer.get_vocab().items():
            token_str = tokenizer.decode(token_id).strip()

            if token_str == "" or (
                all(c.isdigit() or c == "." for c in token_str)
                and token_str.count(".") <= 1
            ):
                self.allowed_mask[token_id] = True

    def __call__(self, _, scores):
        # check that dimensions match
        if scores.shape[-1] != len(self.allowed_mask):
            if scores.shape[-1] < len(self.allowed_mask):
                # remove the last tokens from the mask
                self.allowed_mask = self.allowed_mask[: scores.shape[-1]]
            else:
                # add 0s to the end of the mask
                self.allowed_mask = torch.cat(
                    [self.allowed_mask, torch.zeros(scores.shape[-1] - len(self.allowed_mask), dtype=torch.bool)]
                )
        # take only the scores that correspond to the length of allowed_mask
        mask = self.allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")

        return scores


class LiteralStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        logits_processor: LogitsWarper,
        prompt_length: int,
        literal: List[str],
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.logits_processor = logits_processor
        self.potential_tokens = tokenizer(literal, return_tensors="pt", padding=True)["input_ids"]
        self.logits_processor = logits_processor
        # get the first token of each row of the potential tokens tensor
        self.logits_processor.allowed_tokens = [x[0].item() for x in self.potential_tokens]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        #decoded = self.tokenizer.decode(
        #    input_ids[0][self.prompt_length :], skip_special_tokens=True
        #)
        # check how many self.potential_tokens can start with input_ids
        potential_next_options = []
        input_ids = input_ids[0][self.prompt_length :]
        for idx, tokenized_options in enumerate(self.potential_tokens):
            if tokenized_options[0] == input_ids[-1]:
                potential_next_options.append(idx)
        if len(potential_next_options) == 0:
            raise ValueError("No potential next options found")
        if len(potential_next_options) == 1:
            return True
        else:
            self.potential_tokens = [self.potential_tokens[idx][1:] for idx in potential_next_options]
            self.logits_processor.allowed_tokens = [x[0].item() for x in self.potential_tokens]
            return False

class OutputLiteralsTokens(LogitsWarper):
    def __init__(self):
        self.allowed_tokens = []

    def __call__(self, _, scores):
        # create a mask of allowed tokens
        allowed_mask = torch.zeros(scores.shape[-1], dtype=torch.bool)
        allowed_mask[self.allowed_tokens] = True
        allowed_mask = allowed_mask.expand_as(scores)
        scores[~allowed_mask] = -float("inf")
        return scores