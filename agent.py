import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import AgentState
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph

load_dotenv()

model = ChatOpenRouter(
    api_key=os.getenv('API_KEY'),
    model=os.getenv('MODEL'),
    temperature=0,
)

# State
class State(AgentState):
    # a: int
    # b: int
    iteration: int
    
# Node as a function
def get_user_data(_: State) -> State:
    a = int(input('a = '))
    b = int(input('b = '))
    return {
        'a': a,
        'b': b,
    }
    
def modify(state: State) -> State:
    a, b = state['a'], state['b']
    a, b = b, a % b
    return {
        'a': a,
        'b': b
    }

def write(state: State) -> State:
    print('GCD = ', state['a'])
    return {} 

def ask_llm(state: State) -> State:
    user_query = input("What is your query: ")
    user_message = HumanMessage(content=user_query)
    response = model.invoke(state['messages'] + [user_message])
    print('answer: ', response.content)
    
    return {
        "messages": [user_message, response],
        "iteration": state['iteration'] + 1
    }

# Initial State into Graph
graph = StateGraph(State)

# # Add node into Graph
# graph.add_node('get_user_data', get_user_data)
# graph.add_node('modify', modify)
# graph.add_node('write', write)
graph.add_node('ask_llm', ask_llm)
INTERATION_LIMIT = 5

# Add Edge from node to node
graph.add_edge(START, "ask_llm")
graph.add_edge("ask_llm", "ask_llm")
graph.add_conditional_edges(
    'ask_llm',
    lambda state: state['iteration'] < INTERATION_LIMIT,
    {
        True: 'ask_llm',
        False: END
    },
)
# Add Edges to define the order in which the nodes are executed 
'''Edge from A - B meaning Node A run before Node B,
There are also constant node START and END to know the order of execution
'''

# graph.add_edge(START, 'get_user_data')

# Add conditional Node and Edges into graph
'''Conditional Edges is check to determine from a given Node which Node to execute next,
but we don't want to go back to fetching data, only do it once - so we need some "dummy Node"
to identyfy the start of the loop we can go back.  It's not always necessary, but it's important in this specific case
'''
# # Node loop_conditional
# def loop_conditional(_: State) -> State:
#     return {}

# graph.add_node('loop_condition', loop_conditional)
# graph.add_edge('get_user_data', 'loop_condition')

# # Add conditional loop into graph
# graph.add_conditional_edges(
#     'loop_condition',
#     lambda state: state['b'] != 0, {
#         True: 'modify',
#         False: 'write'
#     },
# )
# graph.add_edge('modify', 'loop_condition')
# graph.add_edge('write', END)

# Compilation
workflow = graph.compile()

# Invoke to run the workflow
workflow.invoke({'iteration': 0, 'messages': []}, {'recursion_limit': 100})