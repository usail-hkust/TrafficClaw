"""Prompt helpers shared by simulation entry points."""

from typing import Callable, Optional


_runtime_query_provider: Optional[Callable[[], Optional[str]]] = None


def set_runtime_query_provider(provider: Optional[Callable[[], Optional[str]]]) -> None:
    """Set an optional runtime query provider used at checkpoint prompt time."""
    global _runtime_query_provider
    _runtime_query_provider = provider


def append_user_query(default_prompt: str, user_query: Optional[str]) -> str:
    """Append an optional user instruction without changing the default prompt."""
    queries = []
    if user_query and user_query.strip():
        queries.append(user_query.strip())

    if _runtime_query_provider is not None:
        runtime_query = _runtime_query_provider()
        if runtime_query and runtime_query.strip():
            queries.append(runtime_query.strip())

    if not queries:
        return default_prompt

    return (
        f"{default_prompt.rstrip()}\n\n"
        "Additional user query/instruction:\n"
        f"{chr(10).join(queries)}"
    )
