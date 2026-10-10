from agent import llm

response = llm.invoke(
    "Explain what an AI agent is in one sentence."
)
print(response.content)