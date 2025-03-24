import time
import pandas as pd

# Modules 
from src.parser import url_parser
from src.extractor import Extractor
from src.data_insert import PGVecInsert
from src.document_chunking import chunking_documents
from src.load_embedding_model import load_embedding_model

# PG Store 
from langchain_postgres import PGVector

# 수집 카테고리 설정
category_lst_main = [
    "https://finance.naver.com/research/market_info_list.naver",
    "https://finance.naver.com/research/market_info_list.naver",
    "https://finance.naver.com/research/economy_list.naver",
    "https://finance.naver.com/research/economy_list.naver",
    "https://finance.naver.com/research/debenture_list.naver"
]

category_lst_industry = [
    "https://finance.naver.com/research/company_list.naver",
    "https://finance.naver.com/research/industry_list.naver"
]

# 리스트 페이지 수집
def fetch_list_data(url, page):
    @url_parser(url=url, pages=page)
    def wrapped(response):
        extract = Extractor(response)
        return extract.extract_element()
    return wrapped()

def fetch_plus_list_data(url, page):
    @url_parser(url=url, pages=page)
    def wrapped(response):
        extract = Extractor(response)
        return extract.extract_industry_stockitems()
    return wrapped()

# 상세 본문 수집
def fetch_detail_content(detail_url):
    @url_parser(url=detail_url, pages=1)
    def wrapped(response):
        extract = Extractor(response)
        return extract.detail_page_crawler()
    return wrapped()

# 🚀 메인 실행
def main():
    final_data = []

    print("\n [1] 네이버 리서치 문서 수집 시작")
    for url in category_lst_main:
        list_data = fetch_list_data(url, page=1)
        for item in list_data:
            content = fetch_detail_content(item["link"])
            item["content"] = content
            item["items"] = None  # 종목 없음
            final_data.append(item)
            time.sleep(3)

    for url in category_lst_industry:
        list_data = fetch_plus_list_data(url, page=1)
        for item in list_data:
            content = fetch_detail_content(item["link"])
            item["content"] = content
            final_data.append(item)
            time.sleep(3)

    df = pd.DataFrame(final_data)
    print(f"총 수집 문서 수: {len(df)}건")

    # DB 설정
    db_config = {
        "dbname": "mydatabase",
        "user": "myuser",
        "password": "mypassword",
        "host": "localhost",
        "port": 5432
    }

    # 임베딩 모델 로딩
    embedding_model = load_embedding_model()

    # 벡터스토어 연결
    vectorstore = PGVector(
        connection="postgresql+psycopg2://myuser:mypassword@localhost:5432/mydatabase",
        collection_name="industry_reports",
        embeddings=embedding_model,
        use_jsonb=True
    ) # 요 부분이 내 생각엔 하드코딩 같기도 하고 ... 왜냐면 db_config랑 같이 또 connection 됐잖아

    # 인서트 클래스 생성 및 삽입
    inserter = PGVecInsert(
        db_config=db_config,
        embedding_model=embedding_model,
        vectorstore=vectorstore
    )

    inserter.insert_dataframe(df, chunking_fn=chunking_documents)

    print(f"\n✅ 총 {len(df)}건 처리 및 저장 완료!")

if __name__ == "__main__":
    main()
