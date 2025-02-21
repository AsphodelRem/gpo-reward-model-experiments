import torch
from transformers import AutoTokenizer


class GRMDataCollatorWithPadding:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.label_pad_token_id = -100

    def __call__(self, features):
        batch = self._prepare_batch(features)
        batch["label"] = self._pad_labels(features, batch["input_ids"].shape[1])
        return batch

    def _prepare_batch(self, features):
        merged_features = [
            {
                "input_ids": feature[key],
                "attention_mask": feature[f"attention_mask_{key.split('_')[1]}"],
            }
            for feature in features
            for key in ["input_ids_chosen", "input_ids_rejected"]
        ]
        return self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

    def _pad_labels(self, features, padded_length):
        label_padded = [
            torch.tensor(
                feature[label].tolist()
                + [self.label_pad_token_id] * (padded_length - len(feature[label])),
                dtype=torch.int64,
            ).view(1, -1)
            for feature in features
            for label in ["label_chosen", "label_rejected"]
        ]
        return torch.cat(label_padded, dim=0)
