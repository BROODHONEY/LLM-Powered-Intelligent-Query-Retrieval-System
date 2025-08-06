import subprocess
import json

def evaluate_decision(parsed_query, retrieved_clauses, model_name="mistral"):
    prompt = f"""
You are a policy assistant. Based on the query and relevant policy clauses below, decide whether the insurance claim should be approved.

### Query Info:
- Age: {parsed_query.get("age")}
- Gender: {parsed_query.get("gender")}
- Procedure: {parsed_query.get("procedure")}
- Location: {parsed_query.get("location")}
- Policy Duration: {parsed_query.get("policy_duration")} months

### Relevant Clauses:
{chr(10).join(retrieved_clause["chunk"] for retrieved_clause in retrieved_clauses)}

### Task:
Determine whether the claim should be approved, the payout amount (if applicable), and provide a justification that references the relevant clauses.

Respond in **this exact JSON format**:

{{
  "decision": "approved/rejected",
  "amount": "number or null",
  "justification": "Concise explanation with clause references"
}}
"""

    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )

        output = result.stdout.decode("utf-8").strip()

        # Optional: try to parse JSON if you want structured output
        try:
            json_start = output.find("{")
            json_text = output[json_start:]
            parsed = json.loads(json_text)
            return parsed
        except Exception as e:
            return {
                "decision": "error",
                "amount": None,
                "justification": f"Could not parse JSON. Raw output:\n{output}"
            }

    except subprocess.TimeoutExpired:
        return {
            "decision": "error",
            "amount": None,
            "justification": "Ollama model timed out."
        }
    except Exception as e:
        return {
            "decision": "error",
            "amount": None,
            "justification": f"Error running ollama: {str(e)}"
        }
