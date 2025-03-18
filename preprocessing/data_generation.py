import os
from openai import OpenAI
import pandas as pd
import time
import json
from dotenv import load_dotenv

load_dotenv("C:/Users/user/Desktop/Finance-Model/.refine_env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

prompt_template = """
Context information is below. You are only aware of this context and nothing else.
---------------------

{context}

---------------------
Given this context, generate exactly **{num_questions}** question(s), answer(s), and a detailed explanation.

You are a Teacher/Professor in {domain}. 
Your task is to create exactly {num_questions} diverse question(s) for an upcoming quiz/examination.
The question(s) should fully reflect your understanding of the context.
You must provide:
- A question extracted from the context.
- A detailed explanation (provide reasoning, step by step, but do not reveal chain-of-thought in a raw format).
- A final answer in a complete sentence (in Korean), prefixed with "정답:".

IMPORTANT:
- You MUST only provide factual and accurate information based on the provided context. Do NOT fabricate or assume any information that is not clearly stated in the context.
- You MUST mention the company name '{company_name}' explicitly in both 'question', 'detailed_explanation', and 'answer'. Do NOT use vague expressions like 'the company' or '동사'. You must directly write '{company_name}' when referring to the company.
- Return the result in JSON format containing the keys `instruction`, `question`, `detailed_explanation`, and `answer`.
  - `instruction` is fixed: "다음 Context를 바탕으로 문맥과 수치를 참고하여 문제 해결 과정을 거쳐 정답을 도출하세요."
  - `question` must contain the question text.
  - detailed_explanation: Based on the provided context, include specific numerical values related to {company_name} (e.g., revenue, EPS, growth rate, RPO, QoQ, etc.) and explain in detail the logical thought process on how these figures contributed to deriving the final answer.
  - answer: Include only the final answer, ensuring that {company_name} is mentioned, and present it concisely.
- DO NOT use List or Array in JSON. Each QA pair must be a separate JSON object.
"""

# 1) 데이터 한 건에 대해 OpenAI 호출하는 함수
def call_openai_api(context: str, company_name: str, domain: str = "경제", num_questions: int = 1):
    user_content = prompt_template.format(
        context=context,
        domain=domain,
        num_questions=num_questions,
        company_name=company_name
    )

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": user_content}],
    )

    return response.choices[0].message.content


# 2) 중간 저장 기능이 포함된 데이터 처리 함수
def process_data(df: pd.DataFrame, save_path: str, context_column: str = "상세내용", company_column: str = "회사명", save_every: int = 10):
    results = []
    start_idx = 0

    # 기존 저장된 파일이 있으면 불러오기
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                processed_indexes = {res["index"] for res in results}
                start_idx = max(processed_indexes) + 1 if processed_indexes else 0
                print(f"[INFO] Resuming from index {start_idx}...")
            except json.JSONDecodeError:
                print("[WARNING] 기존 파일이 손상되었거나 비어 있음. 새로운 파일을 만듭니다.")
                results = []
    
    # 새로운 데이터 처리
    for idx, row in df.iterrows():
        if idx < start_idx:
            continue  # 기존에 처리한 데이터 건너뛰기
        
        context = row[context_column]
        company_name = row[company_column]

        try:
            print(f"[INFO] Processing index {idx}, Company: {company_name}...")
            result = json.loads(call_openai_api(context=context, company_name=company_name)[8:-4])
            results.append({
                "index": idx,
                "company_name": company_name,
                "context": context,
                "result": result
            })
        except Exception as e:
            print(f"[ERROR] Failed to process index {idx}: {e}")
            results.append({
                "index": idx,
                "company_name": company_name,
                "context": context,
                "result": None,
                "error": str(e)
            })

        # 일정 개수마다 저장
        if idx % save_every == 0 or idx == len(df) - 1:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"[INFO] Saved progress at index {idx}")

        time.sleep(1)  # API 요청 간격

    return results


if __name__ == "__main__":
    df = pd.read_csv("../data/2024-2025-mirae_asset.csv", encoding="cp949", sep="\t")
    
    output_file = "data_lst_second.json"
    results = process_data(df, save_path=output_file, context_column="상세내용", company_column="회사명", save_every=10)

    print("[INFO] Final results saved.")
