import re
from typing import List 

def filter_irrelevant_chunks(chunks: List[str], min_len: int = 30) -> List[str]:
    seen = set()
    cleaned = []

    for text in chunks:
        text = text.strip()

        if not text or len(text) < min_len:
            continue

        if text in seen:
            continue

        # 키워드가 포함되어 있으면 제거
        if (
            "자료" in text or
            "그림" in text or
            "참고" in text or
            re.match(r"^/ ?\d+$", text) or
            re.match(r"^\| *구분 *\|", text)
        ):
            continue

        seen.add(text)
        cleaned.append(text)

    return cleaned