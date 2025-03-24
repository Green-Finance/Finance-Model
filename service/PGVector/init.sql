CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE industry_reports (
    id SERIAL PRIMARY KEY,             -- 기본 키 
    title TEXT,
    link TEXT,
    pdf TEXT,
    stock TEXT,
    date TEXT,
    item TEXT,
    content TEXT,
    document TEXT,                     -- 청크된 텍스트
    embedding VECTOR(768),             -- 벡터 임베딩
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
