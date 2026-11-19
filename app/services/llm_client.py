
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


def ask_llm(question: str) -> str:

    # 2. Send the question and tool description to Gemini
    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=question)]
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=types.GenerateContentConfig(
            tools=[calculator_tool]
        )
    )

    # 3. Check whether Gemini requested a tool
    if not response.function_calls:
        return response.text or "No response generated."

    function_call = response.function_calls[0]

    print(f"\nGemini requested tool: {function_call.name}")
    print(f"Arguments: {function_call.args}")

    # 4. Python executes the requested function
    if function_call.name == "multiply":
        result = multiply(**function_call.args)
    elif function_call.name == "add":
        result = add(**function_call.args)
    else:
        raise ValueError("Unknown tool requested")

    print(f"Python tool result: {result}")

    # 5. Send the result back to Gemini
    function_response = types.Part.from_function_response(
        name=function_call.name,
        response={"result": result}
    )

    tool_message = types.Content(
        role="user",
        parts=[function_response]
    )

    final_response = client.models.generate_content(
        model=model_name,
        contents=[
            user_message,
            response.candidates[0].content,
            tool_message
        ],
        config=types.GenerateContentConfig(
            tools=[calculator_tool]
        )
    )

    # 6. Return Gemini's final answer
    return final_response.text or "No final response generated."
