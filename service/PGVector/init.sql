CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE industry_reports (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),     -- LangChain이 요구하는 UUID
    collection_name TEXT DEFAULT 'industry_reports',     -- 문서 그룹 지정
    title TEXT,
    link TEXT,
    pdf TEXT,
    stock TEXT,
    date TEXT,
    item TEXT,
    content TEXT,            -- 전체 내용
    document TEXT,           -- 청킹된 조각 (→ page_content)
    embedding VECTOR(768),   -- 임베딩 벡터
    cmetadata JSONB,         -- 메타데이터 (→ metadata)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
