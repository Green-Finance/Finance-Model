from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

class TrainerSetup:
    def __init__(self, model, tokenizer, dataset, max_seq_length=2048, seed=3407):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.trainer = None

    def setup_trainer(self):
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=self.seed,
            output_dir="outputs",
            report_to="none",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # 짧은 시퀀스에 대해 5배 빠른 학습 가능
            args=training_args,
        )
        return self.trainer