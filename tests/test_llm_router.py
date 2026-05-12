from router.llm_router import MIRINDA_SYSTEM_PROMPT


def test_mirinda_system_prompt_names_agent_and_preserves_rag_rules() -> None:
    assert "Mirinda" in MIRINDA_SYSTEM_PROMPT
    assert "live AI assistant" in MIRINDA_SYSTEM_PROMPT
    assert "answer general questions normally" in MIRINDA_SYSTEM_PROMPT
    assert "use the provided document context as the source of truth" in MIRINDA_SYSTEM_PROMPT
    assert "1-3 short sentences" in MIRINDA_SYSTEM_PROMPT
    assert "Do not sound like a report" in MIRINDA_SYSTEM_PROMPT
    assert "Prefer spoken phrasing over resume formatting" in MIRINDA_SYSTEM_PROMPT
    assert "SkippyEd from January to present" in MIRINDA_SYSTEM_PROMPT
    assert "Do not imitate any specific fictional character" in MIRINDA_SYSTEM_PROMPT
