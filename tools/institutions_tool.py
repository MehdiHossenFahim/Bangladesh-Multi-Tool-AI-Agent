"""
InstitutionsDBTool
------------------
Answers questions about Bangladeshi educational / government institutions:
schools, colleges, madrasahs, universities' affiliated colleges, management
type (govt/non-govt), MPO status, education level, and location (division /
district / thana / union).

Backed by data/db/institutions.db (table: institutions).
Source dataset: Mahadih534/Institutional-Information-of-Bangladesh
"""
from langchain_core.tools import Tool

from tools.db_tool_base import SQLDBTool

SCHEMA_DESCRIPTION = """
- name (TEXT): institution name
- eiin_code (INTEGER): official EIIN identifier
- institute_type (TEXT): e.g. School, College, Madrasha, School and College
- division (TEXT): administrative division, e.g. DHAKA, CHATTOGRAM, RAJSHAHI
- district (TEXT): district name, e.g. BARGUNA, DHAKA, GAZIPUR
- thana (TEXT): thana/upazila-level area name
- union_name (TEXT): union name
- mauza_name (TEXT): mauza (lowest revenue unit) name
- area_status (TEXT): e.g. RURAL, UPZILA SADAR MUNICIPALITY
- geographical_status (TEXT): e.g. PLAIN LAND, COASTAL AREA, RIVER SIDE/CHAR
- address (TEXT): free-text address
- post_office (TEXT): post office name
- management_type (TEXT): GOVERNMENT / NON-GOVERNMENT / LOCAL GOVERNMENT / etc.
- mobile (TEXT): contact number
- student_type (TEXT): CO-EDUCATION JOINT / GIRLS / etc.
- education_level (TEXT): e.g. Secondary, Dakhil, Degree (Pass), MBBS (Medical)
- affiliation_status (TEXT): RECOGNIZE / PERMITTED
- mpo_status (TEXT): YES / NO (government subsidy status)
"""

EXAMPLE_QUERIES = """
Q: How many government institutions are in Rajshahi?
SQL: SELECT COUNT(*) FROM institutions WHERE LOWER(district) LIKE LOWER('%rajshahi%') AND LOWER(management_type) LIKE LOWER('%government%') AND LOWER(management_type) NOT LIKE LOWER('%non-government%');

Q: Which institutions in Bangladesh offer medical degrees?
SQL: SELECT name, district, education_level FROM institutions WHERE LOWER(education_level) LIKE LOWER('%medical%') LIMIT 25;

Q: List colleges in Dhaka district.
SQL: SELECT name, address, management_type FROM institutions WHERE LOWER(district) LIKE LOWER('%dhaka%') AND LOWER(institute_type) LIKE LOWER('%college%') LIMIT 25;

Q: How many madrasahs are there in total?
SQL: SELECT COUNT(*) FROM institutions WHERE LOWER(institute_type) LIKE LOWER('%madrasha%');
"""


def build_institutions_tool(llm, db_path: str = "data/db/institutions.db") -> Tool:
    sql_tool = SQLDBTool(
        name="InstitutionsDBTool",
        description=(
            "Use this tool to answer questions about Bangladeshi educational and "
            "government institutions: schools, colleges, madrasahs, universities' "
            "affiliated colleges, management type, MPO status, education level "
            "offered, or counts/lists filtered by division, district, or upazila. "
            "Examples: 'Which universities offer medical degrees?', "
            "'How many government institutions are in Rajshahi?', "
            "'List colleges in Dhaka district.'"
        ),
        db_path=db_path,
        table_name="institutions",
        schema_description=SCHEMA_DESCRIPTION,
        example_queries=EXAMPLE_QUERIES,
        llm=llm,
    )
    return Tool(
        name=sql_tool.name,
        description=sql_tool.description,
        func=sql_tool.run,
    )
