"""
HospitalsDBTool
----------------
Answers questions about Bangladeshi health facilities: hospitals, medical
college hospitals, upazila health complexes, dispensaries, and other DGHS /
private health facilities, filterable by facility type, location, and
public/private status.

Backed by data/db/hospitals.db (table: hospitals).
Source dataset: Mahadih534/all-bangladeshi-hospitals

NOTE: This source dataset lists health FACILITIES (with agency, type, and
location) rather than per-hospital bed counts or doctor headcounts. If a
question needs bed capacity or staffing numbers that aren't in the data, the
tool will say so honestly instead of inventing figures -- route those to the
WebSearchTool or note the data gap to the user.
"""
from langchain_core.tools import Tool

from tools.db_tool_base import SQLDBTool

SCHEMA_DESCRIPTION = """
- facility_id (INTEGER): internal id
- name (TEXT): facility name (English)
- name_bangla (TEXT): facility name (Bangla script)
- facility_code (INTEGER): official facility code
- agency (TEXT): operating agency, e.g. DGHS, MOPA, Private
- facility_type (TEXT): e.g. Medical College Hospital, Upazila Health Complex,
  District Hospital, Union Health Sub Center, Urban Dispensary,
  Postgraduate Institute & Hospital, Private Specialized Hospital
- division (TEXT): administrative division, e.g. Dhaka, Chattogram
- district (TEXT): district name
- city_corporation (TEXT): city corporation name, if applicable
- upazila (TEXT): upazila/thana name
- paurasava (TEXT): municipality name, if applicable
- union_name (TEXT): union name, if applicable
- is_private (INTEGER): 1 if privately operated, 0 if government/public
"""

EXAMPLE_QUERIES = """
Q: How many hospitals are in Dhaka?
SQL: SELECT COUNT(*) FROM hospitals WHERE LOWER(district) LIKE LOWER('%dhaka%') AND LOWER(facility_type) LIKE LOWER('%hospital%');

Q: List top hospitals in Dhaka.
SQL: SELECT name, facility_type, upazila FROM hospitals WHERE LOWER(district) LIKE LOWER('%dhaka%') AND LOWER(facility_type) LIKE LOWER('%hospital%') LIMIT 10;

Q: Which medical college hospitals exist in Chattogram?
SQL: SELECT name, upazila FROM hospitals WHERE LOWER(district) LIKE LOWER('%chattogram%') AND LOWER(facility_type) LIKE LOWER('%medical college hospital%');

Q: How many private hospitals are there?
SQL: SELECT COUNT(*) FROM hospitals WHERE is_private = 1;

Q: List upazila health complexes in Dhaka division.
SQL: SELECT name, district, upazila FROM hospitals WHERE LOWER(division) LIKE LOWER('%dhaka%') AND LOWER(facility_type) LIKE LOWER('%upazila health complex%') LIMIT 25;
"""


def build_hospitals_tool(llm, db_path: str = "data/db/hospitals.db") -> Tool:
    sql_tool = SQLDBTool(
        name="HospitalsDBTool",
        description=(
            "Use this tool to answer questions about Bangladeshi hospitals and "
            "health facilities: medical college hospitals, upazila health "
            "complexes, district hospitals, dispensaries, and private hospitals, "
            "filterable by location (division/district/upazila), facility type, "
            "or public vs private ownership. NOTE: this data does not include "
            "bed-capacity or doctor-count numbers. "
            "Examples: 'How many hospitals are in Dhaka?', "
            "'List medical college hospitals in Chattogram.', "
            "'How many private hospitals are there?'"
        ),
        db_path=db_path,
        table_name="hospitals",
        schema_description=SCHEMA_DESCRIPTION,
        example_queries=EXAMPLE_QUERIES,
        llm=llm,
    )
    return Tool(
        name=sql_tool.name,
        description=sql_tool.description,
        func=sql_tool.run,
    )
