"""Phase 8: the retrieval layer for the travel guides.

    guides/*.md --chunk by heading--> Documents (+ city, section metadata)
                --embed (cached)----> InMemoryVectorStore
    query --embed--> similarity search, FILTERED to one city --> top-k chunks + scores
                --> confidence band decides: answer, answer carefully, or say "not in my guides"

Design choices:
- Structural chunking: split on the guides' own headings, one idea per chunk, and prepend the
  heading path ("Tokyo > Visa and entry") so each chunk's vector carries its context.
- Metadata filtering: restrict to the trip's city BEFORE ranking, so Paris chunks never compete
  with Tokyo ones ("the secret weapon").
- Confidence bands: strong match -> answer; middle -> answer only if the text truly supports it;
  weak -> "not in my guides" (proceed / confirm / stop).
- Cache embeddings: the corpus is embedded once; later runs cost nothing.
"""

import gzip
import hashlib
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter

from travel_planner.config import CACHE_DIR, PROJECT_ROOT
from travel_planner.usage import record

GUIDES_DIR = PROJECT_ROOT / "guides"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PRICE_PER_M = 0.02  # USD per 1M tokens, list price
# Confidence bands on cosine similarity (proceed / confirm / stop), calibrated with
# scripts/phase8_guide.py --calibrate. One cutoff can't separate covered from uncovered questions
# ("can I chew gum" scored 0.35 and is covered; a Shinkansen fare scored 0.40 and isn't), so the
# middle band hands the judgment to the LLM with an explicit instruction to refuse if unsupported.
STOP_BELOW = 0.30
CONFIRM_BELOW = 0.45
TOP_K = 3


class CachedEmbeddings(Embeddings):
    """Wraps any embedding model with a disk cache keyed by (model, text). Logs uncached usage."""

    def __init__(self, inner: Embeddings, model_name: str, cache_dir: Path = CACHE_DIR / "embeddings"):
        self.inner, self.model_name = inner, model_name
        self.cache_dir = cache_dir / model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(text.encode()).hexdigest()[:32]}.json"

    def _load(self, text: str) -> list[float] | None:
        """Plain JSON (the app), gzip (the committed eval cache), or a copy from TRAVEL_CACHE_SEED_DIR."""
        path = self._path(text)
        gz = path.with_suffix(".json.gz")
        seed = os.environ.get("TRAVEL_CACHE_SEED_DIR")
        seeded = Path(seed) / "embeddings" / self.model_name / path.name if seed else None
        if gz.exists():
            return json.loads(gzip.decompress(gz.read_bytes()))
        if path.exists():
            return json.loads(path.read_text())
        if seeded and seeded.exists():
            vector = json.loads(seeded.read_text())
            self._save(text, vector)
            return vector
        return None

    def _save(self, text: str, vector: list[float]) -> None:
        if os.environ.get("TRAVEL_CACHE_DIR"):  # the committed eval cache: compress
            self._path(text).with_suffix(".json.gz").write_bytes(gzip.compress(json.dumps(vector).encode(), mtime=0))
        else:
            self._path(text).write_text(json.dumps(vector))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing = []
        for i, text in enumerate(texts):
            vectors[i] = self._load(text)
            if vectors[i] is None:
                missing.append(i)
        if missing:
            fresh = self.inner.embed_documents([texts[i] for i in missing])
            for i, vector in zip(missing, fresh):
                self._save(texts[i], vector)
                vectors[i] = vector
            est_tokens = sum(len(texts[i]) for i in missing) // 4
            record("openai", model=self.model_name, input_tokens=est_tokens, output_tokens=0,
                   cost_usd=round(est_tokens * EMBEDDING_PRICE_PER_M / 1_000_000, 6))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_chunks(guides_dir: Path = GUIDES_DIR) -> list[Document]:
    """One chunk per '##' section, tagged with city + section, with the heading path prepended."""
    splitter = MarkdownHeaderTextSplitter([("#", "city"), ("##", "section")])
    chunks = []
    for path in sorted(guides_dir.glob("*.md")):
        for doc in splitter.split_text(path.read_text()):
            city, section = doc.metadata.get("city"), doc.metadata.get("section")
            if not (city and section):
                continue
            chunk_id = f"{path.stem}/{section.lower().replace(' ', '-')}"
            chunks.append(Document(
                id=chunk_id,
                page_content=f"{city} > {section}\n{doc.page_content}",
                metadata={"city": city, "section": section, "chunk_id": chunk_id, "source": path.name},
            ))
    return chunks


class GuideIndex:
    def __init__(self, embeddings: Embeddings, chunks: list[Document] | None = None):
        self.store = InMemoryVectorStore(embeddings)
        self.chunks = chunks if chunks is not None else load_chunks()
        self.store.add_documents(self.chunks)

    def search(self, query: str, city: str | None = None, k: int = TOP_K) -> list[tuple[Document, float]]:
        """Top-k chunks with cosine similarity, filtered to one city when given."""
        city_filter = (lambda doc: doc.metadata["city"] == city) if city else None
        return self.store.similarity_search_with_score(query, k=k, filter=city_filter)


def default_index() -> GuideIndex:
    from langchain_openai import OpenAIEmbeddings

    return GuideIndex(CachedEmbeddings(OpenAIEmbeddings(model=EMBEDDING_MODEL), EMBEDDING_MODEL))


def confidence_band(best_score: float) -> str:
    if best_score < STOP_BELOW:
        return "stop"
    return "confirm" if best_score < CONFIRM_BELOW else "proceed"
