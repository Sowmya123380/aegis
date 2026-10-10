
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from tools import read_document, extract_important_parts

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    api_key=os.getenv("GEMINI_API_KEY"),
)

tools = [read_document, extract_important_parts]
tool_map = {tool.name: tool for tool in tools}
agent_llm = llm.bind_tools(tools)


def run_agent(question):
    messages = [
        HumanMessage(content=(
            "You are Aegis, an AI Document Processing Agent. "
            "Help users summarize documents and extract important "
            "information. Use the available tools when a file path "
            "is provided. Never invent document contents.\n\n"
            + question
        ))
    ]

    for _ in range(5):
        response = agent_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for call in response.tool_calls:
            tool_result = tool_map[call["name"]].invoke(call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=call["id"],
                )
            )

    return "Stopped after 5 tool-calling rounds."


if __name__ == "__main__":
    print("Aegis Document Agent")
    print("Enter a file path and your request.")
    question = input("Request: ")
    print("\nAegis:", run_agent(question))