from pydantic import BaseModel
from openai import OpenAI
import json
import os
from datasets import Dataset, load_dataset
import time 
from dotenv import load_dotenv
import pandas as pd 

load_dotenv("../.env")

with open("C:/Users/user/Desktop/Finance-Model/data/cleaned_data.json", 'r', encoding="utf-8") as f:
    dataset = json.load(f)

save_path = "../data/finance_CoT.json"

df = pd.DataFrame(dataset)
dataset = Dataset.from_pandas(df)

# 1. OpenAI client 초기화
client = OpenAI()

# 2. CoT 데이터셋 Pydantic 모델 정의 (키 변경)
class CoTDataset(BaseModel):
    instruction: str
    question: str
    complex_cot: str
    answer: str

# 3. 프롬프트 (새로운 키 이름 반영)
prompt = """
Context information is below. You are only aware of this context and nothing else.
---------------------

{context}

---------------------
Given this context, generate exactly **{num_questions}** question(s), answer(s), and chain-of-thought reasoning.

You are a Teacher/Professor in {domain}. 
Your task is to create exactly {num_questions} diverse question(s) for an upcoming quiz/examination.
The question(s) should fully reflect your understanding of the context.
You must provide:
- A question extracted from the context.
- A detailed chain-of-thought (complex_cot) explaining your reasoning process.".
- A final answer in a complete sentence (in Korean)".

IMPORTANT:
- Return the result in JSON format containing the keys `instruction`, `question`, `complex_cot`, and `answer`.
  - `instruction` is fixed: "다음 Context를 바탕으로 문제 해결 과정을 거쳐 정답을 도출하세요."
  - `question` must contain the question text.
  - `complex_cot` must include the full reasoning process.
  - `answer` must include only the final answer".
- DO NOT use List or Array in JSON. Each QA pair must be a separate JSON object.

## Example Output:
```json
{{
    "instruction": "다음 Context를 바탕으로 문맥과 수치를 참고하여 문제 해결 과정을 거쳐 정답을 도출하세요.",
    "question": "배출권을 보유한 기업들이 가격 상승 시 판매를 고려할 수 있나요?",
    "detailed_explanation": "기업은 감축 노력을 통해 초과 배출권을 보유한 상태이며, 배출권 가격 상승은 보유 배출권의 단가를 높여 추가 수익 창출 가능성을 제공한다.",
    "answer": "정답: 배출권을 보유한 기업들은 가격이 상승할 경우 초과 배출권을 판매하여 수익을 창출할 수 있습니다. 예를 들어, 배출권 가격이 톤당 30,000원에서 50,000원으로 상승하면 100톤의 초과 배출권을 보유한 기업은 약 5,000만 원의 수익을 올릴 수 있습니다."
}}
"""



# 4. 데이터 생성 함수 정의
def generate_cot_dataset(context: str, domain: str = "경제", num_questions: int = 3):
    filled_prompt = prompt.format(context=context, domain=domain, num_questions=num_questions)

    # OpenAI structured output (Pydantic 기반)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant generating Chain of Thought (CoT) datasets."},
            {"role": "user", "content": filled_prompt}
        ],
        response_format=CoTDataset  # Pydantic 기반 응답 파싱
    )

    # 여러 개가 올 수 있으므로 choices로 받음
    structured_outputs = [choice.message.parsed for choice in completion.choices]
    return structured_outputs

def formatting_prompts_func(examples):
    instructions = examples["input"]  # 지시사항을 가져옵니다.
    outputs = examples["output"]  # 출력값을 가져옵니다.
    texts = []  # 포맷팅된 텍스트를 저장할 리스트입니다.
    for instruction, output in zip(instructions, outputs):
        # output이 리스트일 경우 join()을 사용하여 하나의 문자열로 합칩니다.
        if isinstance(output, list):
            output_text = " ".join(output)
        else:
            output_text = output
        text = instruction + " " + output_text  # 문자열로 합칩니다.
        texts.append(text)
    return {"text": texts}

def save_to_json(data, filename=save_path):
    if not data:
        print("⚠️ [WARNING] 저장할 데이터가 없습니다!")
        return  # 데이터가 없으면 저장하지 않음

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # ✅ 경로가 없으면 생성

    # ✅ 기존 파일 로드
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ JSON 파일이 손상되었습니다. 새로 생성합니다.")
                existing_data = []
    else:
        existing_data = []

    # ✅ CoTDataset 객체를 `dict`로 변환 후 저장
    converted_data = [item.dict() for item in data]  # ✅ 여기서 `dict()` 변환 추가
    
    existing_data.extend(converted_data)

    # ✅ JSON 파일 저장
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    print(f"✅ [INFO] 데이터 저장 완료! (총 {len(existing_data)}개)")


# ✅ 실행 (dataset을 for 루프로 순차적으로 처리하며 저장)
if __name__ == "__main__":
    # ✅ 기존 데이터 개수 설정
    resume_index = 3082  
    total_examples = len(df)  # ✅ 원본 데이터 개수 확인

    print(f"🔄 [INFO] 기존 데이터 {resume_index}개 처리됨. {resume_index + 1}번째부터 실행.")

    # ✅ 데이터셋 포맷팅 (이미 처리된 부분 제외)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.select(range(resume_index, total_examples))  # ✅ 원본 개수 기준으로 선택

    results = []  # ✅ 임시 저장 리스트
    save_every = 5  # ✅ 5개마다 저장

    try:
        for idx, example in enumerate(dataset, start=resume_index):  # ✅ 기존 개수부터 시작
            try:
                context = example["text"]  # 현재 데이터의 context 선택
                print(f"🔹 [{idx+1}/{total_examples}] 데이터 처리 중...")  # ✅ 총 데이터 개수 기준으로 진행률 표시

                # ✅ CoT 데이터 생성
                cot_data = generate_cot_dataset(context=context,domain="경제")

                if cot_data:
                    results.extend(cot_data)  # ✅ 결과를 임시 리스트에 추가
                else:
                    print("⚠️ 데이터가 생성되지 않았습니다. 다음 항목으로 진행.")

                # ✅ 일정 개수마다 저장
                if (idx + 1) % save_every == 0 or idx == total_examples - 1:
                    save_to_json(results)
                    print(f"✅ [INFO] 저장 완료 (Index: {idx})")
                    results = []  # ✅ 임시 리스트 초기화

                time.sleep(1)  # ✅ API 요청 간격

            except KeyboardInterrupt:
                print("\n🚪 [프로그램 종료] 키보드 인터럽트 발생. 종료합니다.")
                save_to_json(results)  # ✅ 종료 전 데이터 저장
                break
            except Exception as e:
                print(f"⚠️ [예외 발생] {str(e)}. 다음 항목으로 진행합니다.")

    except Exception as e:
        print(f"🚨 [치명적 오류] {str(e)}. 프로그램을 종료합니다.")
        save_to_json(results)  # ✅ 프로그램이 비정상 종료될 때도 데이터 저장
