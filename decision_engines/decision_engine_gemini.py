import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Load API key from environment variable or directly assign it here
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-pro")

def evaluate_decision(parsed_query, retrieved_clauses):
    prompt = f"""
You are a policy assistant. Given the user's query details and the relevant policy clauses, make a decision.

### Query Info:
- Age: {parsed_query["age"]}
- Gender: {parsed_query["gender"]}
- Procedure: {parsed_query["procedure"]}
- Location: {parsed_query["location"]}
- Policy Duration: {parsed_query["policy_duration"]} months

### Relevant Clauses:
{chr(10).join(retrieved_clauses)}

### Task:
Determine whether the claim should be approved, the payout amount (if applicable), and provide a justification that references the relevant clauses.

Respond in **this exact JSON format**:

{{
  "decision": "approved/rejected",
  "amount": "number or null",
  "justification": "Concise reason with clause references"
}}
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return {
            "decision": "error",
            "amount": None,
            "justification": f"Error during Gemini call: {str(e)}"
        }
