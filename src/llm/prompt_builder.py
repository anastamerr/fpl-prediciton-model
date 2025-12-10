"""
Structured prompt builder following Context-Persona-Task layout.
"""

from typing import Any, Dict, List, Optional
import json


FPL_PERSONA = (
    "You are an FPL (Fantasy Premier League) expert assistant specializing in team "
    "recommendations and player analysis. You use only provided knowledge graph data "
    "and avoid fabricating stats."
)

TASK_DIRECTIVE = (
    "Answer the user's question using ONLY the information provided in the context. "
    "If the context is insufficient, say so clearly and avoid guessing. Keep answers "
    "concise and reference relevant stats."
)


class PromptBuilder:
    def __init__(self, persona: str = FPL_PERSONA, task_directive: str = TASK_DIRECTIVE):
        self.persona = persona
        self.task_directive = task_directive

    def build_messages(
        self,
        user_query: str,
        kg_context: Any,
        extra_instructions: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        context_str = self._format_context(kg_context)
        system_prompt = f"{self.persona}\n\nCONTEXT:\n{context_str}\n\nTASK:\n{self.task_directive}"
        if extra_instructions:
            system_prompt += f"\n\nADDITIONAL:\n{extra_instructions}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

    def _format_context(self, context: Any) -> str:
        if context is None:
            return "No context retrieved."
        if isinstance(context, str):
            return context
        if isinstance(context, dict):
            parts = []
            anchor = context.get("anchor_player")
            summary_text = context.get("embedding_summary_text")
            if anchor:
                parts.append(f"Anchor player for similarity: {anchor}")
            if summary_text:
                parts.append(f"Similar players: {summary_text}")
            if context.get("baseline"):
                parts.append(f"Baseline rows: {len(context.get('baseline', []))}")
            if context.get("embedding_hits"):
                parts.append(f"Embedding hits: {len(context.get('embedding_hits', []))} (top shown)")
            try:
                raw = json.dumps(context, indent=2)[:6000]
            except TypeError:
                raw = str(context)
            return "\n".join(parts + ["RAW CONTEXT:", raw])
        try:
            return json.dumps(context, indent=2)[:6000]
        except TypeError:
            return str(context)
