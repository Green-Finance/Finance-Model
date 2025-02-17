from unsloth import FastLanguageModel


def generate_response(alpaca_prompt, model, tokenizer, instruction, additional_context="", max_new_tokens=64):
    """
    모델을 inference 모드로 전환한 후, 주어진 instruction을 이용해 응답을 생성합니다.

    Args:
        alpaca_prompt (str): 프롬프트 템플릿. 예: "### Instruction:\n{}\n### Response:\n{}"
        model: 로드된 모델 객체.
        tokenizer: 로드된 토크나이저 객체.
        instruction (str): 생성할 질문.
        additional_context (str): 프롬프트에 추가로 포함시킬 내용 (없으면 빈 문자열).
        max_new_tokens (int): 생성할 최대 토큰 수.

    Returns:
        list: 생성된 응답 문자열 리스트.
    """
    # 추론 모드 활성화 (한 번만 호출해도 충분합니다)
    model = FastLanguageModel.for_inference(model)

    # 입력값 준비: instruction과 추가 context를 프롬프트에 채워 넣습니다.
    prompt_text = alpaca_prompt.format(instruction, additional_context)
    inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
    ).to("cuda")

    # 모델로부터 응답 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    # 응답 디코딩 (불필요한 토큰 제거)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses


def main():
    # 프롬프트 템플릿 정의
    alpaca_prompt = "### Instruction:\n{}\n### Response:\n{}"

    # 저장된 모델과 토크나이저 불러오기 (./result 디렉토리에 저장되어 있어야 함)
    model, tokenizer = FastLanguageModel.from_pretrained("./result", max_seq_length=2048)

    # 테스트를 위한 질문과 추가 context 설정 (추가 context가 필요 없다면 빈 문자열로 둡니다)
    instruction = "bc카드는 어떤 회사인가요?"
    additional_context = ""  # 필요 시 여기에 추가 정보를 입력하세요.

    # 응답 생성
    responses = generate_response(alpaca_prompt, model, tokenizer, instruction, additional_context)

    # 결과 출력
    print("\n=== 생성된 응답 ===\n")
    for response in responses:
        print(response)


if __name__ == "__main__":
    main()