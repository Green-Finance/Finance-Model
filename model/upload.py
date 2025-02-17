from unsloth import FastLanguageModel


def main():
    # 저장된 모델과 토크나이저 불러오기 (예: "./result" 디렉토리에 저장되어 있음)
    print("Loading model and tokenizer from './result' ...")
    model, tokenizer = FastLanguageModel.from_pretrained("./result", max_seq_length=2048)

    # 업로드할 Hugging Face Hub 리포지토리와 토큰 설정 (자신의 값으로 변경)
    repo_id = "username/adapter-model"  # 예: "myusername/my-adapter-model"

    # 모델과 토크나이저를 Hugging Face Hub에 업로드
    print(f"Uploading model to repository '{repo_id}' ...")
    model.push_to_hub(repo_id)

    print(f"Uploading tokenizer to repository '{repo_id}' ...")
    tokenizer.push_to_hub(repo_id)

    print("Upload complete.")


if __name__ == "__main__":
    main()