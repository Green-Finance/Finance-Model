CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE industry_reports (
    id SERIAL PRIMARY KEY,
    item TEXT,  -- NULL이 필요함 (산업, 종목 분석만 있고 나머지는 없음)
    title TEXT,
    link TEXT,
    pdf TEXT,
    stock TEXT,
    date TEXT,
    content TEXT,
    chunk TEXT,
    embed VECTOR(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
