"""Vector store management for Tarkov quests using ChromaDB with hybrid retrieval."""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import chromadb

from src.models import Quest

logger = logging.getLogger(__name__)


class QuestVectorStore:
    """Manages embedding, indexing, and hybrid retrieval for Tarkov quests."""

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "tarkov_quests",
        quests_json_path: str = "data/processed/quests.json",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.quests_json_path = Path(quests_json_path)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._quest_graph: Dict[str, Quest] = {}
        self._quests_list: List[Quest] = []

        # Auto-load graph if processed data exists
        if self.quests_json_path.exists():
            try:
                self.load_quests_from_json(str(self.quests_json_path))
            except Exception as e:
                logger.warning(f"Could not auto-load quests JSON: {e}")

    def load_quests_from_json(self, json_path: str = "data/processed/quests.json") -> List[Quest]:
        """Load and deserialize quests from the processed JSON dataset."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Processed quests file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        quests = [Quest.model_validate(item) for item in raw_data]
        self._quests_list = quests
        self._build_graph(quests)
        return quests

    def _build_graph(self, quests: List[Quest]):
        """Index quests by normalized name and ID for fast graph traversal and exact matching."""
        self._quest_graph.clear()
        for q in quests:
            self._quest_graph[q.name.lower()] = q
            self._quest_graph[q.quest_id] = q

    def get_prerequisite_chain(self, quest_name: str, max_depth: int = 5) -> List[str]:
        """Traverse upstream prerequisite quests recursively."""
        chain: List[str] = []
        visited = set()

        def _traverse(name: str, depth: int):
            if depth > max_depth:
                return
            normalized = name.lower()
            if normalized in visited:
                return
            visited.add(normalized)

            quest = self._quest_graph.get(normalized)
            if not quest:
                return

            for prereq in quest.prerequisites:
                if prereq not in chain and prereq.lower() != quest_name.lower():
                    chain.append(prereq)
                    _traverse(prereq, depth + 1)

        _traverse(quest_name, 1)
        return chain

    def index_quests(self, quests: List[Quest], batch_size: int = 100) -> int:
        """Embed and upsert quest documents into ChromaDB."""
        self._build_graph(quests)
        self._quests_list = quests
        total = len(quests)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for q in quests:
            ids.append(q.quest_id)
            documents.append(q.to_searchable_text())
            metadatas.append(
                {
                    "name": q.name,
                    "trader": q.trader or "Unknown",
                    "location": q.location or "Unknown",
                    "wiki_url": q.wiki_url,
                    "is_kappa_required": bool(q.is_kappa_required) if q.is_kappa_required is not None else False,
                    "required_player_level": q.required_player_level or 0,
                }
            )

        # Upsert in chunks
        for i in range(0, total, batch_size):
            chunk_ids = ids[i : i + batch_size]
            chunk_docs = documents[i : i + batch_size]
            chunk_meta = metadatas[i : i + batch_size]
            self.collection.upsert(
                ids=chunk_ids,
                documents=chunk_docs,
                metadatas=chunk_meta,
            )

        logger.info(f"Successfully indexed {total} quests into ChromaDB.")
        return total

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        trader_filter: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval (dense vector similarity + lexical exact matching + RRF).
        Guarantees that exact quest mentions (e.g. 'Debut', 'Textile') are strongly boosted,
        while maintaining semantic relevance for thematic questions.
        """
        where_filter = None
        if trader_filter:
            where_filter = {"trader": trader_filter}

        # Step 1: Dense retrieval from ChromaDB
        fetch_k = min(max(n_results * 5, 25), 898)
        vector_results = self.collection.query(
            query_texts=[query_text],
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        if not vector_results or not vector_results["ids"] or not vector_results["ids"][0]:
            return []

        if not use_hybrid or not self._quests_list:
            # Fallback to pure vector results
            return self._format_raw_results(vector_results, n_results)

        # Step 2: Reciprocal Rank Fusion (RRF)
        # RRF score = sum(weight / (k + rank))
        k = 60
        rrf_scores: Dict[str, float] = {}

        # Dense vector ranks
        for rank, quest_id in enumerate(vector_results["ids"][0]):
            rrf_scores[quest_id] = rrf_scores.get(quest_id, 0.0) + (1.0 / (k + rank + 1))

        # Lexical / entity matching against quest names and aliases
        query_lower = query_text.lower()
        stop_words = {"what", "are", "the", "for", "to", "in", "do", "i", "need", "unlock", "how", "a", "an", "is", "of", "and"}
        query_tokens = [w for w in re.findall(r"\w+", query_lower) if w not in stop_words]

        lexical_matches: List[tuple[str, float]] = []
        for quest in self._quests_list:
            name_lower = quest.name.lower()
            score = 0.0

            # 1. Exact full title in query (e.g. "debut", "textile - part 1")
            if name_lower in query_lower:
                score += 100.0
            # 2. Main title word match (e.g. "textile" in "what do i need to unlock textile?")
            else:
                main_title_words = [w for w in re.findall(r"\w+", name_lower) if w not in ("part", "a", "the", "in", "to", "for", "of")]
                matches = sum(1 for w in main_title_words if w in query_tokens)
                if matches > 0:
                    score += matches * 20.0

            if score > 0:
                lexical_matches.append((quest.quest_id, score))

        lexical_matches.sort(key=lambda x: x[1], reverse=True)

        # Add lexical ranks to RRF (weighted higher for entity recognition)
        for rank, (quest_id, _) in enumerate(lexical_matches[:25]):
            rrf_scores[quest_id] = rrf_scores.get(quest_id, 0.0) + (2.5 / (k + rank + 1))

        # Step 3: Sort by fused RRF score
        top_ids = sorted(rrf_scores.keys(), key=lambda q_id: rrf_scores[q_id], reverse=True)[:n_results]

        # Step 4: Format output
        formatted_results = []
        for q_id in top_ids:
            quest = self._quest_graph.get(q_id)
            if not quest:
                continue

            prereq_chain = self.get_prerequisite_chain(quest.name)
            score = rrf_scores[q_id]

            formatted_results.append(
                {
                    "quest_id": quest.quest_id,
                    "name": quest.name,
                    "trader": quest.trader or "Unknown",
                    "location": quest.location or "Unknown",
                    "wiki_url": quest.wiki_url,
                    "similarity": round(score, 4),
                    "document": quest.to_searchable_text(),
                    "prerequisite_chain": prereq_chain,
                    "quest_obj": quest,
                }
            )

        return formatted_results

    def _format_raw_results(self, vector_results: dict, n_results: int) -> List[Dict[str, Any]]:
        """Format pure vector search results."""
        formatted = []
        for idx in range(min(n_results, len(vector_results["ids"][0]))):
            quest_id = vector_results["ids"][0][idx]
            metadata = vector_results["metadatas"][0][idx]
            document = vector_results["documents"][0][idx]
            distance = vector_results["distances"][0][idx] if vector_results.get("distances") else 0.0
            similarity = round(1.0 - distance, 4) if distance is not None else 1.0

            quest_name = metadata.get("name", "")
            prereq_chain = self.get_prerequisite_chain(quest_name)

            formatted.append(
                {
                    "quest_id": quest_id,
                    "name": quest_name,
                    "trader": metadata.get("trader"),
                    "location": metadata.get("location"),
                    "wiki_url": metadata.get("wiki_url"),
                    "similarity": similarity,
                    "document": document,
                    "prerequisite_chain": prereq_chain,
                }
            )
        return formatted
