import os

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
checkpointer = InMemorySaver()

model = ChatOpenRouter(
    api_key=os.getenv('API_KEY'),
    model=os.getenv('MODEL'),
    temperature=0,
)

# Search model = LLM + Search Tool
model_with_search = model.bind_tools([DuckDuckGoSearchResults()])

# # Write tools
# @tool
# def get_weather(city: str) -> str:
#     """Get the weather in a city"""
#     return f"It's rainy in {city}"

class State(AgentState):
    iteration: int

def ask_llm(state: State) -> State:
    user_input = input("User: ")
    user_message = HumanMessage(content=user_input)
    response = model_with_search.invoke(state['messages'] + [user_message])
    return {
        "messages": [user_message, response],
        "iteration": state['iteration'] + 1
    }

def show_answer(state: State) -> State:
    print("Answer ", state["messages"][-1].content)
    return {
        'iteration': state['iteration']
    }
    
def sum_up_search(state: State) -> State:
    answer_mess = model.invoke(state['messages'])
    return {
        'messages': [answer_mess],
    }

# Initial Graph    
graph = StateGraph(State)

# Add Node
graph.add_node('ask_llm', ask_llm)
graph.add_node('web_search', ToolNode(tools=[DuckDuckGoSearchResults()]))
graph.add_node('show_answer', show_answer)
graph.add_node('sum_up_search', sum_up_search)

# Add Edge Node to Node
graph.add_edge(START, 'ask_llm')
graph.add_conditional_edges(
    'ask_llm',
    tools_condition,
    {
        'tools': 'web_search',
        END: 'show_answer'
    }
)

# Add Edge Node to Node
graph.add_edge('web_search', 'sum_up_search')
graph.add_edge('sum_up_search', 'show_answer')
graph.add_conditional_edges(
    'show_answer',
    lambda state: state['iteration'] < 10,
    {
        True: 'ask_llm',
        False: END
    }
)

workflow = graph.compile()

with open('graph.png', 'wb') as f:
    f.write(workflow.get_graph().draw_mermaid_png())

workflow.invoke({'iteration': 0}, {'recursion_limit': 100})


# config = {
#     "configurable": {
#         "thread_id": "1",
#     }
# }

# agents = create_agent(
#     model = model,
#     tools=[get_weather],
#     checkpointer=checkpointer
# )

# while True:
#     user_input = input("User: ")
#     new_state  = agents.invoke(
#         {'messages': [HumanMessage(content=user_input)]}, 
#         config
#     )
#     answer = new_state['messages'][-1].content
#     print('Agent: ', answer)
