import re

def parse_query(query):
    parsed = {
        "age" : None,
        "gender": None,
        "procedure": None,
        "location": None,
        "policy_duration": None,
    }
    
    match = re.search(r'(\d+)[\s-]*year[-\s]*old\s*(male|female)', query.lower())
    if match:
        parsed["age"] = int(match.group(1))
        parsed["gender"] = match.group(2)

    proc_match = re.search(r'(knee surgery|surgery|operation)', query.lower())
    if proc_match:
        parsed["procedure"] = proc_match.group(1)

    loc_match = re.search(r'in\s+(\w+)', query.lower())
    if loc_match:
        parsed["location"] = loc_match.group(1)

    duration_match = re.search(r'(\d+)[\s-]*month', query.lower())
    if duration_match:
        parsed["policy_duration"] = int(duration_match.group(1))

    return parsed
