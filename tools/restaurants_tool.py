"""
RestaurantsDBTool
------------------
Answers questions about Bangladeshi restaurants: ratings, number of reviews,
location (address / lat-lon), affluence tier, and a best-effort cuisine
guess derived from the restaurant name.

Backed by data/db/restaurants.db (table: restaurants).
Source dataset: Mahadih534/Bangladeshi-Restaurant-Data

NOTE: The source dataset (scraped from Google Places) has no explicit
cuisine field. `cuisine_guess` is derived from keyword matching on the
restaurant name at ingest time (see scripts/build_databases.py) and is a
heuristic, not verified data -- the tool should be transparent about that
when it matters.
"""
from langchain_core.tools import Tool

from tools.db_tool_base import SQLDBTool

SCHEMA_DESCRIPTION = """
- place_id (TEXT): Google Places identifier
- name (TEXT): restaurant/eatery name (English and/or Bangla)
- latitude (REAL), longitude (REAL): coordinates
- rating (REAL): average rating out of 5 (0 if unrated)
- number_of_reviews (REAL): number of reviews (NULL if none)
- affluence (REAL): a relative price/affluence tier signal (1-4), often NULL
- address (TEXT): free-text address, usually ending in an area/city name
- cuisine_guess (TEXT): heuristic cuisine category derived from the name,
  one of: biryani, chinese, fast_food, sweets_bakery, tea_stall,
  traditional_bangladeshi, cafe, unknown
"""

EXAMPLE_QUERIES = """
Q: Find restaurants in Chattogram serving biryani.
SQL: SELECT name, address, rating FROM restaurants WHERE LOWER(address) LIKE LOWER('%chattogram%') AND (cuisine_guess = 'biryani' OR LOWER(name) LIKE LOWER('%biryani%')) LIMIT 25;

Q: What are the highest rated restaurants in Dhaka?
SQL: SELECT name, address, rating, number_of_reviews FROM restaurants WHERE LOWER(address) LIKE LOWER('%dhaka%') AND rating > 0 ORDER BY rating DESC, number_of_reviews DESC LIMIT 10;

Q: How many restaurants are in the dataset for Swarupkathi?
SQL: SELECT COUNT(*) FROM restaurants WHERE LOWER(address) LIKE LOWER('%swarupkathi%');

Q: List Chinese restaurants with more than 50 reviews.
SQL: SELECT name, address, rating, number_of_reviews FROM restaurants WHERE cuisine_guess = 'chinese' AND number_of_reviews > 50 ORDER BY number_of_reviews DESC LIMIT 25;
"""


def build_restaurants_tool(llm, db_path: str = "data/db/restaurants.db") -> Tool:
    sql_tool = SQLDBTool(
        name="RestaurantsDBTool",
        description=(
            "Use this tool to answer questions about Bangladeshi restaurants: "
            "ratings, review counts, location/address, and cuisine (a "
            "best-effort guess based on the restaurant name, since the source "
            "data has no verified cuisine field). "
            "Examples: 'Find restaurants in Chattogram serving biryani.', "
            "'What are the highest rated restaurants in Dhaka?', "
            "'List Chinese restaurants with more than 50 reviews.'"
        ),
        db_path=db_path,
        table_name="restaurants",
        schema_description=SCHEMA_DESCRIPTION,
        example_queries=EXAMPLE_QUERIES,
        llm=llm,
    )
    return Tool(
        name=sql_tool.name,
        description=sql_tool.description,
        func=sql_tool.run,
    )
