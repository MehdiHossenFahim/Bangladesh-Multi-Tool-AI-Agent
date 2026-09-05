"""
WebSearchTool
--------------
General-knowledge web search for anything NOT covered by the three
structured databases: definitions, policy questions, cultural context,
current events, etc.

Provider priority:
  1. Tavily (TAVILY_API_KEY set)         -- best quality, LLM-oriented search
  2. DuckDuckGo (no API key required)    -- automatic fallback, works offline
     of any paid service so the project still runs with zero search API keys.

Swap in SerpAPI / Bing by adding another branch in `_search()` if preferred.
"""
import os

from langchain_core.tools import Tool


def _search_tavily(query: str) -> str:
    from langchain_community.tools.tavily_search import TavilySearchResults

    tool = TavilySearchResults(max_results=5)
    results = tool.invoke({"query": query})
    lines = []
    for r in results:
        title = r.get("title") or r.get("url", "")
        content = r.get("content", "")
        url = r.get("url", "")
        lines.append(f"- {title}: {content}\n  Source: {url}")
    return "\n".join(lines) if lines else "No results found."


def _search_duckduckgo(query: str) -> str:
    from duckduckgo_search import DDGS

    lines = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"- {title}: {body}\n  Source: {href}")
    return "\n".join(lines) if lines else "No results found."


def _search(query: str) -> str:
    if os.getenv("TAVILY_API_KEY"):
        try:
            return _search_tavily(query)
        except Exception as e:  # noqa: BLE001
            return f"[Tavily search failed: {e}] Falling back to DuckDuckGo.\n" + _search_duckduckgo(query)
    return _search_duckduckgo(query)


def build_web_search_tool(llm) -> Tool:
    def run(query: str) -> str:
        raw_results = _search(query)
        prompt = f"""A user asked: "{query}"

Here are raw web search results:
{raw_results}

Using ONLY this information, write a concise, well-organized answer (2-5
sentences, or a short list if appropriate). Mention if the information may
be incomplete or if sources disagree. Do not fabricate facts not present in
the search results."""
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    return Tool(
        name="WebSearchTool",
        description=(
            "Use this tool for general knowledge questions NOT answerable from "
            "the institutions/hospitals/restaurants databases: definitions, "
            "government policy, healthcare policy, history, culture, current "
            "events, or anything requiring up-to-date web information. "
            "Examples: 'What is the healthcare policy of Bangladesh?', "
            "'What is the role of DGHS in Bangladesh?'"
        ),
        func=run,
    )
