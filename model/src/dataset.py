from datasets import load_dataset

class DatasetPreparer:
    def __init__(self, dataset_name="yahma/alpaca-cleaned", split="train"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

    @staticmethod
    def _alpaca_prompt_format(instruction, input_text, output, eos_token):
        alpaca_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{}\n\n"
            "### Input:\n{}\n\n"
            "### Response:\n{}"
        )
        return alpaca_prompt.format(instruction, input_text, output) + eos_token

    def _formatting_prompts_func(self, examples, eos_token):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, inp, output in zip(instructions, inputs, outputs):
            text = self._alpaca_prompt_format(instruction, inp, output, eos_token)
            texts.append(text)
        return {"text": texts}

    def prepare_dataset(self, tokenizer):
        eos_token = tokenizer.eos_token
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        self.dataset = self.dataset.map(lambda examples: self._formatting_prompts_func(examples, eos_token), batched=True)
        return self.dataset