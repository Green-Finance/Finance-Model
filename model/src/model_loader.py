from unsloth import FastLanguageModel

class ModelLoader:
    def __init__(self, model_name="unsloth/Meta-Llama-3.1-8B", max_seq_length=2048, dtype=None, load_in_4bit=True, random_state=3407):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.random_state = random_state
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        return self.model, self.tokenizer

    def prepare_peft_model(self):
        if self.model is None:
            raise ValueError("먼저 load_model()을 호출하여 모델을 로딩하세요.")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # 추천: 8, 16, 32, 64, 128 중 하나 선택
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # 0이면 최적화됨
            bias="none",     # "none"이면 최적화됨
            use_gradient_checkpointing="unsloth",  # 매우 긴 컨텍스트 지원
            random_state=self.random_state,
            use_rslora=False,
            loftq_config=None,
        )
        return self.model, self.tokenizer