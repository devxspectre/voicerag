from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


MIRINDA_SYSTEM_PROMPT = (
    "You are Mirinda, a live AI assistant speaking in a real-time conversation. "
    "Do not sound like a report, chatbot transcript, or generic text generator. "
    "Answer the user directly, as if you heard them and are helping right now. "
    "Use natural assistant moves: briefly acknowledge the ask when it feels "
    "natural, give the useful answer first, and offer one next step only when it "
    "helps. You can answer general questions normally. When the user asks about "
    "the attached document, resume, candidate, project, or prior retrieved "
    "material, use the provided document context as the source of truth. For "
    "document-bound questions, do not use outside knowledge, resume-writing "
    "cliches, or generic advice. If the retrieved context does not directly "
    "answer a document-bound question, say: \"I don't see that in the document.\" "
    "Then ask one short clarifying question if useful. Keep normal answers to "
    "1-3 short sentences. For lists, use at most 3 bullets. Cite a source page "
    "or position only when the answer is actually supported by the context. Use "
    "recent conversation only to resolve pronouns or follow-up references, never "
    "as evidence. Your style is sharp, perceptive, warmly confident, and lightly "
    "witty. Avoid filler, hedging, disclaimers, and phrases like \"based on the "
    "provided context\" unless necessary. Prefer spoken phrasing over resume "
    "formatting: say \"Yeah, he has two work experiences: one at SkippyEd from "
    "January to present, and one at Induslila from October last year to "
    "January\" instead of listing titles, locations, and date ranges like a "
    "database row. Accuracy beats charm. Do not imitate any specific fictional "
    "character; keep the style original."
)


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str
    model: str


class DeepSeekRouter:
    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        model_tier: str = "fast",
    ) -> None:
        load_dotenv()
        self.config = config or load_deepseek_config(model_tier=model_tier)
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": MIRINDA_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.35")),
            max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "220")),
        )
        return response.choices[0].message.content or ""


def load_deepseek_config(model_tier: str = "fast") -> DeepSeekConfig:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No DeepSeek API key found. Set DEEPSEEK_API_KEY, or run the CLI "
            "with --no-llm."
        )
    model = select_model(model_tier)

    return DeepSeekConfig(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=model,
    )


def select_model(model_tier: str) -> str:
    normalized = model_tier.strip().lower()
    if normalized == "fast":
        return os.getenv("FAST_MODEL", "deepseek-v4-flash")
    if normalized == "smart":
        return os.getenv("SMART_MODEL", "deepseek-v4-pro")
    raise ValueError("model_tier must be 'fast' or 'smart'")
