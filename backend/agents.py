import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
import os 
load_dotenv()

from langgraph.graph import StateGraph , END
from langchain_tavily import TavilySearch
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage, ChatMessage
import operator
from langgraph.checkpoint.sqlite import SqliteSaver

# Backend setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class TaskRequest(BaseModel):
    task: str

memory_ctx = SqliteSaver.from_conn_string(":memory:")
memory = memory_ctx.__enter__() 
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: list[str]
    revision_number: int
    max_revisions: int
    
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash", temperature=0.2)

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays of very huge length.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""



class Queries(BaseModel):
    queries: List[str]

tavily = TavilySearch(tavily_api_key= os.getenv("TAVILY_SEARCH_API_KEY"))

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content= PLAN_PROMPT),
        HumanMessage(content = state['task'])
    ]
    response = model.invoke(messages)
    return {"plan" : response.content}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content = RESEARCH_PLAN_PROMPT),
        HumanMessage(content = state['plan'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.invoke({"query": q, "num_results": 2})
        for r in response.get("results", []):
            content.append(r['content'])
    return {"content": content}


def generation_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content = f"{state['task']} \n\n Here is my  plan: \n\n {state['plan']}"
    )
    messages = [
        SystemMessage(content = WRITER_PROMPT.format(content = content)),
        user_message
    ]
    response = model.invoke(messages)
    return {
        "draft" : response.content,
        "revision_number": state.get("revision_number", 1) +1
        }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content  = REFLECTION_PROMPT),
        HumanMessage(content = state["draft"])
    ]
    response = model.invoke(messages)
    return {"critique" : response.content}
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content = RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content = state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries: 
        response = tavily.invoke({"query": q, "num_results": 2})
        for r in response.get("results", []):
            content.append(r["content"])
    return {"content": content}


def should_continue(state: AgentState):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"
builder = StateGraph(AgentState)
builder.add_node("planner" , plan_node)
builder.add_node("research_plan" , research_plan_node)
builder.add_node("generator", generation_node)
builder.add_node("reflect" , reflection_node)
builder.add_node("research_critique" , research_critique_node)
builder.set_entry_point("planner")
builder.add_conditional_edges(
    "generator", 
    should_continue,
    {
        END : END,
        "reflect" : "reflect"
    }
)

builder.add_edge("planner" , "research_plan")
builder.add_edge("research_plan" , "generator")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generator")
graph=builder.compile(checkpointer=memory)


@app.post("/api/essay")
async def generate_essay(request: TaskRequest):
    # Initialize the LangGraph state
    initial_state: AgentState = {
        "task": request.task,        # <-- comes from frontend
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 1,
        "max_revisions": 2
    }
    thread = {"configurable" : {"thread_id" : "2"}}
    # Run the LangGraph workflow
    output = graph.invoke(initial_state,thread)
    return {"output": output.get("draft")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)