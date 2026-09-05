"""
agent.py
--------
Main entry point for the Bangladesh Multi-Tool AI Agent.

Builds a LangChain tool-calling AgentExecutor wired up to four tools:
  - InstitutionsDBTool
  - HospitalsDBTool
  - RestaurantsDBTool
  - WebSearchTool

The agent's LLM decides, per-query, which tool(s) to call based on each
tool's description -- e.g. "how many hospitals are in Dhaka" routes to
HospitalsDBTool, while "what is the healthcare policy of Bangladesh" routes
to WebSearchTool.

Usage:
    python agent.py                       # interactive CLI chat
    python agent.py --query "..."         # single one-off question
"""
import argparse
import os
import sys

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import (
    build_hospitals_tool,
    build_institutions_tool,
    build_restaurants_tool,
    build_web_search_tool,
)

load_dotenv(override=True)

DB_DIR = os.path.join(os.path.dirname(__file__), "data", "db")

SYSTEM_PROMPT = """You are a helpful multi-tool assistant for questions about Bangladesh.

You have access to three structured SQLite databases (via tools) and a web
search tool:
- InstitutionsDBTool: schools, colleges, madrasahs, government/educational institutions.
- HospitalsDBTool: hospitals and health facilities.
- RestaurantsDBTool: restaurants, ratings, cuisine, locations.
- WebSearchTool: general knowledge, definitions, policy, and cultural context
  not covered by the databases above.

Routing rules:
1. If the question asks for counts, lists, or specific records about
   institutions, hospitals, or restaurants in Bangladesh, use the matching
   database tool.
2. If the question is about general knowledge, definitions, government
   policy, or anything not in the databases, use WebSearchTool.
3. If a question needs both (e.g. a database fact plus context), call both
   tools and combine the results.
4. Always answer in clear, natural language. If a database tool reports no
   data or an unsupported field (e.g. bed-capacity numbers that don't exist
   in the hospitals data), say so honestly instead of guessing.
5. Be concise but complete: use short lists for multiple records.
"""


def build_llm():
    """
    Builds the chat model used both for tool routing (the agent) and for
    text-to-SQL / summarization inside each tool.

    Defaults to Groq (LLM_PROVIDER=groq) because it gives a free API key
    with just an email/Google sign-in -- no credit card required, and it's
    fast. Other options (all free-tier, no card, unless noted):

      LLM_PROVIDER=groq      -> free key at https://console.groq.com/keys
      LLM_PROVIDER=gemini    -> free key at https://aistudio.google.com/apikey
      LLM_PROVIDER=ollama    -> fully local, no signup/key at all, needs
                                 https://ollama.com installed + a model pulled
                                 (e.g. `ollama pull llama3.1`)
      LLM_PROVIDER=anthropic -> requires ANTHROPIC_API_KEY (billing/card typically required)
      LLM_PROVIDER=openai    -> requires OPENAI_API_KEY (billing/card typically required)
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=0,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        temperature=0,
    )


def build_agent() -> AgentExecutor:
    llm = build_llm()

    tools = [
        build_institutions_tool(llm, db_path=os.path.join(DB_DIR, "institutions.db")),
        build_hospitals_tool(llm, db_path=os.path.join(DB_DIR, "hospitals.db")),
        build_restaurants_tool(llm, db_path=os.path.join(DB_DIR, "restaurants.db")),
        build_web_search_tool(llm),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def check_databases_exist() -> None:
    missing = [
        name
        for name in ("institutions.db", "hospitals.db", "restaurants.db")
        if not os.path.exists(os.path.join(DB_DIR, name))
    ]
    if missing:
        print(
            f"Missing database(s): {', '.join(missing)}.\n"
            "Run this first:  python scripts/build_databases.py --source sample\n"
            "(or --source raw after running scripts/download_datasets.py)"
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Run a single query and exit.")
    args = parser.parse_args()

    check_databases_exist()
    executor = build_agent()

    if args.query:
        result = executor.invoke({"input": args.query})
        print("\n" + result["output"])
        return

    print("Bangladesh Multi-Tool AI Agent. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = executor.invoke({"input": query})
        print(f"\nAgent: {result['output']}\n")


if __name__ == "__main__":
    main()
