import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from .document_chunking import chunking_documents


class PGVecInsert:
    def __init__(self, db_config: dict):
        self.db_config = db_config

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def insert_dataframe(self, df: pd.DataFrame, table_name: str = "industry_reports"):
        insert_query = f"""
        INSERT INTO {table_name} (title, link, pdf, stock, date, content, embed, item, chunk)
        VALUES %s
        """

        records = []

        print(f"🔄 전체 문서 수: {len(df)}")

        for idx, row in df.iterrows():
            print(f"\n📄 [{idx+1}] 처리 중: {row['title'][:30]}...")

            # content: 요약 텍스트 정제
            content_str = "\n".join(row["content"]) if isinstance(row["content"], list) else row["content"]

            try:
                print(f"📥 PDF 임베딩 중... ({row['pdf']})")
                chunks = chunking_documents(row["pdf"])  # List[(text, vector)]
                print(f"✅ 총 청크 수: {len(chunks)}")
            except Exception as e:
                print(f"❌ PDF 처리 실패: {row['pdf']} / 에러: {e}")
                continue

            for i, (chunk_text, vector) in enumerate(chunks):
                records.append((
                    row["title"],
                    row["link"],
                    row["pdf"],
                    row["stock"],
                    row["date"],
                    content_str,
                    vector,
                    row["items"],
                    chunk_text
                ))

                if i == 0:
                    print(f"📎 첫 번째 청크: {chunk_text[:40]}...")

        print(f"\n📦 최종 삽입 레코드 수: {len(records)}")

        # DB 삽입
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, insert_query, records)
                conn.commit()
            print("✅ PGVector DB에 정상 삽입 완료!")
        except Exception as e:
            print(f"❌ DB 삽입 중 오류 발생: {e}")

