# 🏦 Finance-Model

> SBA 서울경제진흥원 도봉캠퍼스 2기 주역들의 러닝메이트 작품  
> 프로젝트명: `GreenFinance`

---

## 👨‍👩‍👧‍👦 조원 소개

| 이름 | 프로필 | GitHub |
|------|--------|--------|
| 지용욱 | <img src="https://avatars.githubusercontent.com/u/52349219?v=4" width="100"/> | [GitSkyBlue](https://github.com/GitSkyBlue) |
| 송진희 | <img src="https://avatars.githubusercontent.com/u/139064340?v=4" width="100"/> | [jinheesong](https://github.com/jinheesong) |
| 박제형 | <img src="https://avatars.githubusercontent.com/u/192846476?v=4" width="100"/> | [PJH-02](https://github.com/PJH-02) |
| 정수인 | <img src="https://avatars.githubusercontent.com/u/192847666?v=4" width="100"/> | [sooin1516717](https://github.com/sooin1516717)|
| 김호준 | <img src="https://avatars.githubusercontent.com/u/192846581?s=64&v=4" width="100"/> | [megashot](https://github.com/megashot)|
| 황의철(팀장) | <img src="https://avatars.githubusercontent.com/u/109947779?v=4" width="100"/> | [UICHEOL-HWANG](https://github.com/UICHEOL-HWANG) |


## 🎯 목표

- 금융/산업 보고서 데이터를 기반으로 **강력한 RAG(Retrieval-Augmented Generation)** 시스템을 구축
- 향후 확장 가능한 **Agent 기반 투자 분석 도우미**로 발전시킬 수 있는 토대 설계

---

## 🧠 모델 구조 개요

### 1. 문서 수집 및 정제
- PDF 형식의 금융 보고서 크롤링
- **Docling**을 활용한 마크다운 정제
- 문서 내용을 자연어 기준으로 **청크 단위 분할**

### 2. 임베딩 처리 및 저장
- 임베딩 모델: `jhgan/ko-sroberta-multitask`
- 청크별로 임베딩 후 **PostgreSQL + pgvector**에 저장

```sql
-- LangChain 호환 pgvector 테이블 구조
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE industry_reports (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_name TEXT DEFAULT 'industry_reports',
    title TEXT,
    link TEXT,
    pdf TEXT,
    stock TEXT,
    date TEXT,
    item TEXT,
    content TEXT,
    document TEXT,             -- 청크된 텍스트
    embedding VECTOR(768),     -- 벡터
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔧 초기 트러블슈팅 (실제 겪은 문제와 해결 방법)

| 이슈 | 해결 방법 |
|------|------------|
| `embedding` 값이 pgvector와 LangChain에서 따로 관리됨 | `add_documents()`를 쓰면 중복 임베딩 발생함을 인지 → `add_embeddings()`로 직접 임베딩 처리 및 저장 방식으로 전환 |
| 기존 테이블과 LangChain 포맷이 달라 검색 실패 | 테이블 구조를 `embedding`, `document`, `cmetadata`로 리팩터링하여 LangChain 포맷에 맞춤 |
| `get_relevant_documents()`가 문서 0건 반환 | `collection_name` 누락 또는 `use_jsonb=True` 설정 미적용 → 필수 인자 및 설정값 반영하여 해결 |
| LangChain과 SQL 병행 유지 어려움 | 단일 테이블 구조에서 유지/관리 어려움 발생 → **원본 테이블(`industry_reports`)과 langchain PGVector 전용 테이블(`langchain_pg_embedding`)을 분리**하여 2중 관리 체계로 전환 |
| vectorstore 검색 대상 문서가 부실하거나 의미 없는 경우 포함됨 | 저장 전 `page_content` 필터링 로직 추가 (길이 제한 및 특수문자만 포함된 청크 제거)로 불필요한 문서 삽입 방지 |
| 벡터 삽입 후 검색 불가 현상 발생 | LangChain의 PGVector는 내부적으로 필수 필드 (`document`, `embedding`, `cmetadata`) 및 `collection_name`을 사용 → 컬럼 명칭 및 구조를 명확히 맞춰 해결 |


---

## 🔍 검색 시스템 구조 (RAG 전 단계)

- LangChain의 `PGVector`를 직접 연결해 벡터 기반 문서 검색
- `.as_retriever()`로 문서 검색 → 추론 파이프라인 설계 가능
- 필요 시 `ContextualCompressionRetriever` + `CrossEncoder`로 리랭킹 확장 가능

---

## 🤖 Agent 확장을 위한 준비

- 각 리서치 문서를 메타데이터 기준으로 필터링 가능 (`stock`, `item`, `date`)
- Agent가 필요한 경우: `Tool + Function Call` 방식으로 LangChain에 연동
- 향후 FastAPI, LangGraph 기반 Agent 구축 예정 (현재 `Corrective Agent` 구축 완료)

---

## 🧩 프로젝트 구성 요소

| 모듈 | 설명 |
|------|------|
| `PGVecInsert` | 정제된 문서 및 벡터를 DB에 삽입하는 커스텀 클래스 |
| `chunking_documents` | PDF 청크 단위 분리 및 임베딩 전처리 |
| `Docling` | 마크다운 기반의 문서 정제 엔진 |
| `PGVector` | LangChain 기반 벡터 검색기 |
| `FastAPI ` (계획) | 사용자 인터페이스 및 API |

---

## ✨ 향후 계획

- 🔍 RAG 기반 요약/응답 시스템 정식 구축
- 🧠 LangGraph 기반 Agent 연결
- 📊 사용자 관심 종목 기반 알림 서비스 연동
- 🧪 모델 성능 평가 및 리랭커 최적화

---
