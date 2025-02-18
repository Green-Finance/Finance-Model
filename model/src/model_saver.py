class ModelSaver:
    def __init__(self, trainer, save_dir="outputs"):
        """
        trainer: SFTTrainer 객체
        save_dir: 모델과 토크나이저를 저장할 디렉토리
        """
        self.trainer = trainer
        self.save_dir = save_dir

    def save(self):
        # trainer 내부의 save_model() 메서드를 이용하여 모델 저장
        self.trainer.save_model(self.save_dir)
        # 토크나이저도 함께 저장
        self.trainer.tokenizer.save_pretrained(self.save_dir)
        print(f"모델과 토크나이저가 '{self.save_dir}'에 저장되었습니다.")