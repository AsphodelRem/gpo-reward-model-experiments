import os

import torch
import torch.nn.functional as F
from transformers import Trainer

from src.utils.model_utils import get_trainable_weights


class GRMRewardTrainer(Trainer):
    def __init__(self, **kwargs):
        self.reference_free = kwargs.pop("reference_free", True)
        self.reference_model = kwargs.pop("reference_model", None)
        self.sft_only = kwargs.pop("sft_only", True)
        self.no_logsigmoid_sft = kwargs.pop("no_logsigmoid_sft", False)
        self.weight_ratio = kwargs.pop("weight_ratio", 0.01)
        self.beta = kwargs.pop("beta", 0.1)
        self.label_pad_token_id = -100
        self.use_lora = kwargs.pop("use_lora", True)
        self.info_to_save = kwargs.pop("info_to_save", {})

        super().__init__(**kwargs)

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """
        Compute the log probabilities of the given labels under the given logits.
        Returns a tensor containing the average/sum log probabilities.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits and labels must have the same shape in batch and sequence dimensions."
            )

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id
        labels[labels == self.label_pad_token_id] = 0  # Dummy token

        per_token_logps = torch.gather(
            logits.log_softmax(dim=-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        return (per_token_logps * loss_mask).sum(dim=-1)

    def compute_loss(self, model, inputs, return_outputs=False):
        logits, _, rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        if not self.reference_free:
            with torch.no_grad():
                ref_logits = self.reference_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )[0]

        bsz = rewards.size(0)
        jidx, kidx = torch.arange(0, bsz, 2), torch.arange(1, bsz, 2)
        reward_loss = -F.logsigmoid(rewards[jidx] - rewards[kidx]).mean()

        if self.weight_ratio > 0:
            logps = self.get_batch_logps(logits, inputs["label"])
            pi_logratios = logps[jidx]

            if self.sft_only:
                dpo_loss = (
                    -pi_logratios.mean()
                    if self.no_logsigmoid_sft
                    else -F.logsigmoid(self.beta * pi_logratios).mean()
                )
            else:
                pi_logratios -= logps[kidx]
                ref_logratios = (
                    torch.tensor(0.0)
                    if self.reference_free or self.sft_only
                    else self.get_batch_logps(ref_logits, inputs["label"])[jidx]
                    - self.get_batch_logps(ref_logits, inputs["label"])[kidx]
                )

                device = rewards.device
                dpo_loss = -F.logsigmoid(
                    self.beta * (pi_logratios.to(device) - ref_logratios.to(device))
                ).mean()

            loss = self.weight_ratio * dpo_loss + (1 - self.weight_ratio) * reward_loss
        else:
            loss = reward_loss

        return (loss, {}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            logits, _, rewards = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            logps = self.get_batch_logps(logits, inputs["label"])

            if self.reference_free:
                dpo_logp_diff = logps
            else:
                ref_logits = self.reference_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )[0]
                dpo_logp_diff = logps - self.get_batch_logps(
                    ref_logits, inputs["label"]
                )

        return None, dpo_logp_diff.reshape(-1, 2), rewards.reshape(-1, 2)

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save and self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)

            model.config.update(
                {
                    "vhead_layer_type": self.info_to_save["layer_type"],
                    "vhead_num_neurons": self.info_to_save["num_neurons"],
                    "vhead_num_layers": self.info_to_save["num_layers"],
                }
            )

            state_dict = get_trainable_weights(
                model.base_model.model if self.use_lora else model
            )
            model.base_model.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=True
            ) if self.use_lora else model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=False
            )
            self.tokenizer.save_pretrained(output_dir)
