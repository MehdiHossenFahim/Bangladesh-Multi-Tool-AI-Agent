"""
db_tool_base.py
----------------
Shared logic for every database-backed tool in this project:

1. Connect to a SQLite database.
2. Given a natural-language question, ask the LLM to write a single
   read-only SQL query against the known schema.
3. Execute that query safely (SELECT-only, row-limited).
4. Ask the LLM to turn the raw rows into a short, natural-language answer.

Each concrete tool (InstitutionsDBTool, HospitalsDBTool, RestaurantsDBTool)
just supplies: db path, table name, schema description, and a few example
questions -> SQL pairs (few-shot) to keep generated SQL reliable.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

MAX_ROWS_RETURNED = 25


class UnsafeSQLError(ValueError):
    pass


def extract_sql(text: str) -> str:
    """Pull a bare SQL statement out of an LLM response that may contain
    markdown fences or explanatory text."""
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1) if fence_match else text
    candidate = candidate.strip().strip(";").strip()
    return candidate


def enforce_read_only(sql: str) -> str:
    """Raise if the generated SQL looks like it could mutate the database."""
    lowered = sql.strip().lower()
    if not lowered.startswith("select"):
        raise UnsafeSQLError(f"Refusing to run non-SELECT SQL: {sql!r}")
    forbidden = ("insert", "update", "delete", "drop", "alter", "attach", "pragma", "create")
    if any(f" {kw} " in f" {lowered} " for kw in forbidden):
        raise UnsafeSQLError(f"Refusing to run SQL containing a forbidden keyword: {sql!r}")
    return sql


def add_row_limit(sql: str, limit: int = MAX_ROWS_RETURNED) -> str:
    if re.search(r"\blimit\s+\d+", sql, re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {limit}"


@dataclass
class SQLDBTool:
    name: str
    description: str
    db_path: str
    table_name: str
    schema_description: str
    example_queries: str
    llm: Any  # a LangChain chat model with .invoke()

    def _generate_sql(self, question: str) -> str:
        prompt = f"""You are a SQLite expert. Given the schema below, write ONE
read-only SQL query (SELECT only) that answers the user's question.
Return ONLY the SQL, no explanation, no markdown fences.

Table: {self.table_name}
Schema:
{self.schema_description}

Example questions and matching SQL:
{self.example_queries}

Rules:
- Use only the "{self.table_name}" table.
- Use LIKE with wildcards for fuzzy text/location matching (case-insensitive
  via LOWER(column) LIKE LOWER('%value%')).
- Never write INSERT/UPDATE/DELETE/DROP/ALTER.
- Always add a LIMIT clause (<= {MAX_ROWS_RETURNED}) unless the question
  asks for a COUNT/aggregate.

Question: {question}
SQL:"""
        response = self.llm.invoke(prompt)
        raw_sql = response.content if hasattr(response, "content") else str(response)
        sql = extract_sql(raw_sql)
        sql = enforce_read_only(sql)
        sql = add_row_limit(sql)
        return sql

    def _run_sql(self, sql: str) -> tuple[list[str], list[tuple]]:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return columns, rows
        finally:
            conn.close()

    def _summarize(self, question: str, sql: str, columns: list[str], rows: list[tuple]) -> str:
        if not rows:
            return (
                f"I ran a query against the {self.table_name} data but found no matching "
                f"records for: \"{question}\"."
            )
        preview = "\n".join(
            ", ".join(f"{col}={val}" for col, val in zip(columns, row)) for row in rows[:MAX_ROWS_RETURNED]
        )
        prompt = f"""A user asked: "{question}"

This SQL query was run against the Bangladesh {self.table_name} database:
{sql}

Raw results ({len(rows)} row(s), showing up to {MAX_ROWS_RETURNED}):
{preview}

Write a concise, natural-language answer for the user based ONLY on this
data. Use a short list or table-like formatting if there are multiple
records. If a count/aggregate was requested, state the number clearly."""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def run(self, question: str) -> str:
        try:
            sql = self._generate_sql(question)
        except UnsafeSQLError as e:
            return f"I couldn't safely answer that from the {self.table_name} database ({e})."
        try:
            columns, rows = self._run_sql(sql)
        except sqlite3.Error as e:
            return f"The generated SQL failed against the {self.table_name} database: {e}\nSQL was: {sql}"
        return self._summarize(question, sql, columns, rows)
