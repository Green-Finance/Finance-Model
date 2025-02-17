from datasets import load_dataset

class DatasetPreparer:
    def __init__(self, dataset_name="yahma/alpaca-cleaned", split="train"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

    @staticmethod
    def _alpaca_prompt_format(instruction, output, eos_token):
        alpaca_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{}\n\n"
            "### Response:\n{}"
        )
        return alpaca_prompt.format(instruction, output) + eos_token

    def _formatting_prompts_func(self, examples, eos_token):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = self._alpaca_prompt_format(instruction, output, eos_token)
            texts.append(text)
        # 디버그용 전체 텍스트 출력
        print(texts[0])
        return {"text": texts}

    def prepare_dataset(self, tokenizer):
        eos_token = tokenizer.eos_token
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        self.dataset = self.dataset.map(lambda examples: self._formatting_prompts_func(examples, eos_token), batched=True)
        return self.dataset