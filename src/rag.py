"""RAG pipeline integrating retrieval and LLM generation for Tarkov quests."""

import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from src.vectorstore import QuestVectorStore

load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert Escape from Tarkov game assistant.
Your job is to provide accurate, concise, and helpful answers to players asking about in-game quests.

Guidelines:
1. Always base your answers ONLY on the provided quest context. If the context does not contain the answer, state that honestly.
2. When answering "what do I need to unlock X", list any prerequisite quests, trader loyalty levels, and player level requirements found in the context.
3. When answering about quest objectives or rewards, provide a clear, bulleted list.
4. Always provide an explicit markdown citation linking back to the source wiki page at the end of your answer, for example:
   Source: [Quest Name](Wiki URL)
5. Keep your tone direct, knowledgeable, and formatted with clean markdown.
"""


class TarkovRAG:
    """End-to-end Retrieval-Augmented Generation system for Tarkov quests."""

    def __init__(
        self,
        vector_store: Optional[QuestVectorStore] = None,
        model_name: Optional[str] = None,
    ):
        self.vector_store = vector_store or QuestVectorStore()
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")

        self._openai_client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

    @property
    def has_llm_client(self) -> bool:
        """Check if an LLM client is configured with an active API key."""
        return self._openai_client is not None

    def build_prompt_context(self, retrieved_quests: List[Dict[str, Any]]) -> str:
        """Format retrieved quests and prerequisite chains into LLM context."""
        context_blocks = []
        for idx, item in enumerate(retrieved_quests, 1):
            prereq_text = ""
            if item.get("prerequisite_chain"):
                prereq_text = f"\nPrerequisite Chain: {' -> '.join(item['prerequisite_chain'])}"

            block = (
                f"--- CONTEXT DOCUMENT {idx} ---\n"
                f"{item['document']}"
                f"{prereq_text}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def answer_question(
        self,
        question: str,
        n_results: int = 3,
        trader_filter: Optional[str] = None,
        force_offline: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant quest context and generate an answer (or format offline summary).
        """
        # 1. Retrieve relevant quests
        retrieved = self.vector_store.query(
            query_text=question,
            n_results=n_results,
            trader_filter=trader_filter,
            use_hybrid=True,
        )

        if not retrieved:
            return {
                "question": question,
                "answer": "No relevant quests found matching your question.",
                "sources": [],
                "retrieved": [],
                "mode": "none",
            }

        sources = [{"name": r["name"], "url": r["wiki_url"], "trader": r["trader"]} for r in retrieved]

        # 2. Check if LLM generation is available
        if not self.has_llm_client or force_offline:
            # Fallback to structured offline summary card
            offline_answer = self._generate_offline_summary(question, retrieved)
            return {
                "question": question,
                "answer": offline_answer,
                "sources": sources,
                "retrieved": retrieved,
                "mode": "offline",
            }

        # 3. Call LLM
        context = self.build_prompt_context(retrieved)
        user_message = (
            f"Quest Context Information:\n{context}\n\n"
            f"Player Question: {question}\n\n"
            f"Please answer the player's question using the context above. Cite the source quest wiki URL."
        )

        try:
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            )
            llm_text = response.choices[0].message.content.strip()
            return {
                "question": question,
                "answer": llm_text,
                "sources": sources,
                "retrieved": retrieved,
                "mode": "llm",
            }
        except Exception as e:
            logger.error(f"LLM API call failed: {e}. Falling back to offline context summary.")
            offline_answer = self._generate_offline_summary(question, retrieved)
            return {
                "question": question,
                "answer": f"[Notice: LLM call error ({e}). Showing direct context summary]\n\n{offline_answer}",
                "sources": sources,
                "retrieved": retrieved,
                "mode": "offline_fallback",
            }

    @staticmethod
    def _generate_offline_summary(question: str, retrieved: List[Dict[str, Any]]) -> str:
        """Create a clean markdown summary directly from retrieved quest documents."""
        lines = []
        top_quest = retrieved[0]
        lines.append(f"### Relevant Quest: [{top_quest['name']}]({top_quest['wiki_url']})")
        lines.append(f"- **Trader:** {top_quest['trader']}")
        if top_quest.get("location"):
            lines.append(f"- **Location:** {top_quest['location']}")
        if top_quest.get("prerequisite_chain"):
            lines.append(f"- **Prerequisite Chain:** {' → '.join(top_quest['prerequisite_chain'])}")

        lines.append("\n**Details from wiki:**")
        lines.append(top_quest["document"])

        lines.append(f"\n\n**Source:** [{top_quest['name']}]({top_quest['wiki_url']})")
        return "\n".join(lines)
