import json
import operator
import os
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI


class State(TypedDict):
    messages: Annotated[list[dict], operator.add]


def add_numbers(a: int, b: int) -> int:
    return a + b


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]

TOOL_FUNCTIONS = {"add_numbers": add_numbers}
client = OpenAI()


def agent(state: State):
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=state["messages"],
        tools=TOOLS,
    )
    return {"messages": [response.choices[0].message.model_dump(exclude_none=True)]}


def tools(state: State):
    messages = []
    for call in state["messages"][-1].get("tool_calls", []):
        name = call["function"]["name"]
        args = json.loads(call["function"].get("arguments", "{}"))
        result = TOOL_FUNCTIONS[name](**args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": str(result),
            }
        )
    return {"messages": messages}


def next_step(state: State):
    return "tools" if state["messages"][-1].get("tool_calls") else END


workflow = StateGraph(State)
workflow.add_node("agent", agent)
workflow.add_node("tools", tools)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", next_step, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What is 2 + 3?"}]}
    )
    print(result["messages"][-1]["content"])
