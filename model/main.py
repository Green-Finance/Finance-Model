from src.model_loader import ModelLoader
from src.dataset import DatasetPreparer
from src.train import TrainerSetup
from src.model_saver import ModelSaver
# Modules

def main():
    model_loader = ModelLoader(model_name="beomi/Llama-3-Open-Ko-8B-Instruct-preview",
                               max_seq_length=2048,
                               dtype=None,
                               random_state=3407)

    model, tokenizer = model_loader.load_model()
    model, tokenizer = model_loader.prepare_peft_model()

    dataset_preparer = DatasetPreparer(dataset_name="BCCard/BCCard-Finance-Kor-QnA")
    dataset = dataset_preparer.prepare_dataset(tokenizer)

    # 3. Trainer 설정
    trainer_setup = TrainerSetup(model, tokenizer, dataset, max_seq_length=2048, seed=3407)
    trainer = trainer_setup.setup_trainer()

    trainer.train()

    # 5. 모델 저장
    saver = ModelSaver(trainer, save_dir="outputs")
    saver.save()

if __name__ == "__main__":
    main()