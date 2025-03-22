import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from .document_chunking import chunking_documents
import json 


class PGVecInsert:
    def __init__(self, db_config: dict):
        self.db_config = db_config

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def insert_dataframe(self, df: pd.DataFrame, table_name: str = "industry_reports"):
        insert_query = f"""
        INSERT INTO {table_name} (
            collection_name, title, link, pdf, stock, date, item,
            content, document, embedding, cmetadata
        ) VALUES %s
        """

        records = []
        print(f"🔄 전체 문서 수: {len(df)}")

        for idx, row in df.iterrows():
            print(f"\n📄 [{idx+1}] 처리 중: {row['title'][:30]}...")

            content_str = "\n".join(row["content"]) if isinstance(row["content"], list) else row["content"]

            try:
                print(f"📥 PDF 임베딩 중... ({row['pdf']})")
                chunks = chunking_documents(row["pdf"])  # List of (chunk_text, vector)
                print(f"✅ 총 청크 수: {len(chunks)}")
            except Exception as e:
                print(f"❌ PDF 처리 실패: {row['pdf']} / 에러: {e}")
                continue
            
            # 청깅된 데이터, 벡터 데이터 - > 메타데이터로 매핑 langchain Vector search로 보기 위해 
            for i, (chunk_text, vector) in enumerate(chunks):
                metadata = {
                    "title": row["title"],
                    "link": row["link"],
                    "pdf": row["pdf"],
                    "stock": row["stock"],
                    "date": row["date"],
                    "item": row["items"]
                }

                records.append((
                    "industry_reports",          # collection_name
                    row["title"],
                    row["link"],
                    row["pdf"],
                    row["stock"],
                    row["date"],
                    row["items"],
                    content_str,
                    chunk_text,                 # document
                    vector,                     # embedding (VECTOR(768))
                    json.dumps(metadata)        # cmetadata (JSONB)
                ))

                if i == 0:
                    print(f"📎 첫 번째 청크: {chunk_text[:40]}...")

        print(f"\n📦 최종 삽입 레코드 수: {len(records)}")