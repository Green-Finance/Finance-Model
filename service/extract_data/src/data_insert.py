import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from typing import Optional, Any, List, Tuple


class PGVecInsert:
    def __init__(
        self,
        db_config: dict,
        embedding_model: Any,
        vectorstore: Optional[Any] = None
    ):
        self.db_config = db_config
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def insert_dataframe(self, df: pd.DataFrame, chunking_fn: Any, table_name: str = "industry_reports"):
        insert_query = f'''
        INSERT INTO {table_name} (
            title, link, pdf, stock, date, item,
            content, document, embedding
        ) VALUES %s
        '''

        print(f"🔄 전체 문서 수: {len(df)}")

        for idx, row in df.iterrows():
            print(f"\\n📄 [{idx+1}] 처리 중: {row['title'][:30]}...")
            content_str = "\\n".join(row["content"]) if isinstance(row["content"], list) else row["content"]

            try:
                chunks: List[Tuple[str, List[float]]] = chunking_fn(row["pdf"], self.embedding_model)
                print(f"✅ 총 청크 수: {len(chunks)}")
            except Exception as e:
                print(f"❌ PDF 처리 실패: {row['pdf']} / 에러: {e}")
                continue

            records = []
            texts, vectors, metadatas = [], [], []

            for i, (chunk_text, vector) in enumerate(chunks):
                records.append((
                    row["title"],
                    row["link"],
                    row["pdf"],
                    row.get("stock"),
                    row["date"],
                    row.get("items"),
                    content_str,
                    chunk_text,
                    vector
                ))

                texts.append(chunk_text)
                vectors.append(vector)
                metadatas.append({
                    "title": row["title"],
                    "link": row["link"],
                    "date": row["date"]
                })

            # 1. DB 저장
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        execute_values(cur, insert_query, records)
                    conn.commit()
                print(f"✅ [{idx+1}] DB 저장 완료: {len(records)}개 청크")
            except Exception as e:
                print(f"❌ DB 삽입 실패: {e}")

            # 2. LangChain vectorstore 저장
            if self.vectorstore:
                try:
                    self.vectorstore.add_embeddings(texts, vectors, metadatas)
                    print(f"✅ [{idx+1}] LangChain 벡터스토어 저장 완료")
                except Exception as e:
                    print(f"❌ LangChain vectorstore 삽입 실패: {e}")