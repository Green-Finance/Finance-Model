from src.parser import url_parser
from src.extractor import Extractor
from src.data_insert import PGVecInsert

import time
import pandas as pd 



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

def fetch_detail_content(detail_url):
    @url_parser(url=detail_url, pages=1)
    def wrapped(response):
        extract = Extractor(response)
        return extract.detail_page_crawler()
    
    return wrapped()


# 메인 함수 
def main():
    final_data = []
    
    # 산업, 종목 제외한 데이터 수집 
    for url in category_lst_main:
        list_data = fetch_list_data(url, page=1)
        for item in list_data:
            content = fetch_detail_content(item["link"])
            item["content"] = content
            item["items"] = None # item이 없기에 빼고
            final_data.append(item)
            time.sleep(3)
    
    for url in category_lst_industry:
        list_data = fetch_plus_list_data(url, page=1)
        for item in list_data:
            content = fetch_detail_content(item["link"])
            item["content"]= content
            final_data.append(item)
            time.sleep(3)
            
    db_config = {
        "dbname": "mydatabase",
        "user": "myuser",
        "password": "mypassword",
        "host": "localhost",
        "port": 5432
    }

    df = pd.DataFrame(final_data)
    
    inserter = PGVecInsert(
        db_config=db_config
        )
    
    inserter.insert_dataframe(df)
    
    print(f"총 저장된 데이터수 : {len(df)}건")
    

if __name__ == "__main__":
    main()
            