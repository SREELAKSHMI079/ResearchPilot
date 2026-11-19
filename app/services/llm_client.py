
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.tools.calculator import add, multiply

tool_registry = {
    "multiply": multiply,
    "add": add
}

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)


# 1. Describe our calculator tool to Gemini
calculator_function = types.FunctionDeclaration(
    name="multiply",
    description="Multiply two numbers and return the result.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number"
            },
            "b": {
                "type": "number",
                "description": "The second number"
            }
        },
        "required": ["a", "b"]
    }
)
add_function = types.FunctionDeclaration(
    name="add",
    description="Add two numbers and return the result.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number"
            },
            "b": {
                "type": "number",
                "description": "The second number"
            }
        },
        "required": ["a", "b"]
    }
)

calculator_tool = types.Tool(
    function_declarations=[calculator_function,add_function]
)



def ask_llm(question: str, conversation_history: list) -> str:

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=question)]
    )

    # Keep the complete conversation history
    contents = conversation_history+[user_message]
    agent_state = {
    "question": question,
    "tool_calls": [],
    "tool_results": [],
    "steps_completed": 0
}

    # Prevent the agent from running forever
    max_tool_rounds = 5
    tool_round = 0

    while tool_round < max_tool_rounds:

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[calculator_tool]
            )
        )

        # No tool requested: return the final answer
        if not response.function_calls:
            
            final_answer = response.text or "No response generated."

            contents.append(response.candidates[0].content)

            conversation_history.clear()
            conversation_history.extend(contents)

            return final_answer

        
        

        # Execute every requested tool
        for function_call in response.function_calls:

            print(f"\nGemini requested tool: {function_call.name}")
            print(f"Arguments: {function_call.args}")

            tool_function = tool_registry.get(function_call.name)

            if tool_function is None:
                raise ValueError("Unknown tool requested")

            result = tool_function(**function_call.args)
            agent_state["tool_calls"].append(function_call.name)

            agent_state["tool_results"].append(result)

            agent_state["steps_completed"] += 1

            print(f"Python tool result: {result}")

            function_response = types.Part.from_function_response(
                name=function_call.name,
                response={"result": result}
            )

            tool_message = types.Content(
                role="user",
                parts=[function_response]
            )

            contents.append(tool_message)

        tool_round += 1
    print("\n--- Agent State ---")
    print(agent_state)
    return "Maximum tool execution limit reached."

