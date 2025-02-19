import warnings
from datetime import datetime

import comet_ml
import hydra
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from src.utils.init_utils import set_random_seed


def initialize_model_and_tokenizer(config):
    model_config = ModelConfig(**config.model)
    quantization_config = get_quantization_config(model_config)

    model_kwargs = {
        "revision": config.model.model_revision,
        "device_map": get_kbit_device_map()
        if quantization_config is not None
        else None,
        "quantization_config": quantization_config,
        "use_cache": not config.trainer_config.gradient_checkpointing,
        "torch_dtype": None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
        model_max_length=256,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name_or_path,
        num_labels=1,
        trust_remote_code=config.model.trust_remote_code,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    return model, tokenizer


def load_and_prepare_dataset(config: dict):
    dataset = pd.read_csv(config.dataset.dataset_path)
    dataset = Dataset.from_pandas(dataset)

    if "test" not in dataset:
        train_test_split = dataset.train_test_split(
            test_size=config.dataset.train_test_split_ratio
        )
        dataset = DatasetDict(
            {"train": train_test_split["train"], "test": train_test_split["test"]}
        )

    return dataset


@hydra.main(version_base=None, config_path="configs", config_name="base_bradly_terry_qwen_0.5")
def main(config):
    """
    Основной скрипт для обучения модели.
    Инициализирует модель, токенизатор, датасет и запускает обучение.

    Args:
        config (DictConfig): Конфигурация эксперимента.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"qwen-reward-training_{current_time}"

    comet_ml.login(project_name="gpo-reward-model")

    warnings.filterwarnings("ignore", category=UserWarning)
    set_random_seed(config.trainer.seed)

    model, tokenizer = initialize_model_and_tokenizer(config)
    dataset = load_and_prepare_dataset(config)

    def truncate_example(example):
        example["chosen"] = example["chosen"][:512]
        example["rejected"] = example["rejected"][:512]
        return example

    dataset["train"] = dataset["train"].map(truncate_example)

    training_args = RewardConfig(**config.trainer_config)
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=get_peft_config(ModelConfig(**config.model)),
    )

    print("Начало обучения!")
    trainer.train()


if __name__ == "__main__":
    main()
