from datasets import Dataset, DatasetDict
import json
from huggingface_hub import login

def load_and_split_dataset(json_path, test_ratio=0.1):
    """
    JSON 데이터를 로드하고 `train`과 `test`로 나누어 DatasetDict로 변환하는 함수

    Args:
        json_path (str): JSON 파일 경로
        test_ratio (float): 테스트 데이터셋 비율 (기본값: 10%)

    Returns:
        DatasetDict: `train`과 `test`로 나뉜 Hugging Face 데이터셋
    """

    # 1️⃣ JSON 파일 로드
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)  # json.load() 사용

    # 2️⃣ JSON 데이터를 Hugging Face Dataset 변환
    dataset = Dataset.from_list(data)

    # 3️⃣ 데이터 섞기 (셔플)
    dataset = dataset.shuffle(seed=42)

    # 4️⃣ `train`과 `test`로 분할 (10%는 테스트 데이터)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size

    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    # 5️⃣ DatasetDict로 변환
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    return dataset_dict

if __name__ == "__main__":
    login(token="token")
    json_path = "../preprocessing/cleaned_data.json"

    # 데이터셋 로드 및 분할 (test 10%)
    dataset = load_and_split_dataset(json_path, test_ratio=0.1)
    
    dataset.push_to_hub("UICHEOL-HWANG/GreenFinance-Finance_CoT")
