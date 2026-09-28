"""과목별 강의 자료 저장소: Chroma 벡터 검색 + BM25를 RRF로 합친 하이브리드 검색."""

import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.loader import Chunk

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
DB_PATH = os.getenv("CHROMA_PATH", "data/chroma")
RRF_K = 60


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float
    id: str = ""


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def _embed(texts: list[str], kind: str) -> list[list[float]]:
    # e5 계열 모델은 "query: " / "passage: " 접두어를 붙여야 성능이 제대로 나온다
    if "e5" in EMBED_MODEL:
        texts = [f"{kind}: {t}" for t in texts]
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def tokenize(text: str) -> list[str]:
    """BM25용 토크나이저. 한글 단어는 조사 문제를 피하려고 bigram도 함께 쓴다."""
    tokens = []
    for word in re.findall(r"\w+", text.lower()):
        tokens.append(word)
        if len(word) > 2 and re.search(r"[가-힣]", word):
            tokens += [word[i : i + 2] for i in range(len(word) - 1)]
    return tokens


class LectureStore:
    def __init__(self, path: str = DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection("lectures", metadata={"hnsw:space": "cosine"})
        self._bm25: dict[str, tuple] = {}  # 과목별 BM25 인덱스 캐시

    # ---------- 저장 ----------

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.col.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=_embed([c.text for c in batch], "passage"),
                metadatas=[{"subject": c.subject, "source": c.source, "page": c.page} for c in batch],
            )
        for subject in {c.subject for c in chunks}:
            self._bm25.pop(subject, None)

    def delete_file(self, subject: str, source: str) -> int:
        ids = self.col.get(where={"$and": [{"subject": subject}, {"source": source}]}, include=[])["ids"]
        if ids:
            self.col.delete(ids=ids)
            self._bm25.pop(subject, None)
        return len(ids)

    # ---------- 조회 ----------

    def subjects(self) -> dict[str, int]:
        """과목별 파일 수."""
        metas = self.col.get(include=["metadatas"])["metadatas"]
        files = {(m["subject"], m["source"]) for m in metas}
        return dict(sorted(Counter(s for s, _ in files).items()))

    def files(self, subject: str) -> list[str]:
        metas = self.col.get(where={"subject": subject}, include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def file_text(self, subject: str, source: str) -> list[tuple[int, str]]:
        """파일 전체 내용을 페이지 순서대로 반환한다 (요약용)."""
        data = self.col.get(
            where={"$and": [{"subject": subject}, {"source": source}]}, include=["documents", "metadatas"]
        )
        rows = sorted(zip(data["ids"], data["metadatas"], data["documents"]), key=lambda r: (r[1]["page"], r[0]))
        return [(m["page"], doc) for _, m, doc in rows]

    def sample(self, subject: str, n: int) -> list[Hit]:
        """퀴즈용으로 과목 자료에서 무작위 청크를 뽑는다."""
        data = self.col.get(where={"subject": subject}, include=["documents", "metadatas"])
        idx = random.sample(range(len(data["ids"])), min(n, len(data["ids"])))
        return [Hit(data["documents"][i], data["metadatas"][i]["source"], data["metadatas"][i]["page"], 0.0) for i in idx]

    # ---------- 검색 ----------

    def search(self, subject: str, query: str, k: int = 5) -> list[Hit]:
        count = len(self.col.get(where={"subject": subject}, include=[])["ids"])
        if count == 0:
            return []
        n = min(max(k * 3, 10), count)

        fused: dict[str, Hit] = {}
        scores: dict[str, float] = {}
        for hits in (self._vector_search(subject, query, n), self._bm25_search(subject, query, n)):
            for rank, h in enumerate(hits, start=1):
                fused.setdefault(h.id, h)
                scores[h.id] = scores.get(h.id, 0.0) + 1 / (RRF_K + rank)

        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [Hit(fused[i].text, fused[i].source, fused[i].page, scores[i], i) for i in top]

    def _vector_search(self, subject: str, query: str, n: int) -> list[Hit]:
        res = self.col.query(query_embeddings=_embed([query], "query"), n_results=n, where={"subject": subject})
        return [
            Hit(doc, meta["source"], meta["page"], 1 - dist, id_)
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def _bm25_search(self, subject: str, query: str, n: int) -> list[Hit]:
        if subject not in self._bm25:
            data = self.col.get(where={"subject": subject}, include=["documents", "metadatas"])
            self._bm25[subject] = (BM25Okapi([tokenize(d) for d in data["documents"]]), data)
        bm25, data = self._bm25[subject]

        scores = bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            Hit(data["documents"][i], data["metadatas"][i]["source"], data["metadatas"][i]["page"], float(scores[i]), data["ids"][i])
            for i in top
            if scores[i] > 0
        ]
