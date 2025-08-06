import re

def parse_query(query):
    parsed = {
        "age": None,
        "gender": None,
        "procedure": None,
        "location": None,
        "policy_duration": None,
    }

    # 1. Age and Gender
    match = re.search(r'(\d+)[\s-]*(?:year[-\s]*old)?\s*(male|female|m|f)', query.lower())
    if match:
        parsed["age"] = int(match.group(1))
        gender = match.group(2)
        parsed["gender"] = "male" if gender in ["m", "male"] else "female"

    # 2. Procedure
    proc_match = re.search(r'(knee surgery|surgery|operation|hip replacement|treatment)', query.lower())
    if proc_match:
        parsed["procedure"] = proc_match.group(1)

    # 3. Location
    loc_match = re.search(r'(?:in|at|from)\s+([a-zA-Z]+)', query.lower())
    if loc_match:
        parsed["location"] = loc_match.group(1)

    # 4. Policy Duration
    duration_match = re.search(r'(\d+)[\s-]*month', query.lower())
    if duration_match:
        parsed["policy_duration"] = int(duration_match.group(1))

    return parsed

def build_search_input(parsed):
    parts = []
    if parsed["age"]: parts.append(f"{parsed['age']} year old")
    if parsed["gender"]: parts.append(parsed["gender"])
    if parsed["procedure"]: parts.append(f"procedure: {parsed['procedure']}")
    if parsed["location"]: parts.append(f"in {parsed['location']}")
    if parsed["policy_duration"]: parts.append(f"{parsed['policy_duration']}-month old policy")

    return " ".join(parts)