from typing import Dict, TypedDict, Optional
from fastapi import FastAPI, File
from langgraph.graph import StateGraph, END
from pymongo import MongoClient
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


APP_HOST = "127.0.0.1"
APP_PORT = 8080

load_dotenv()

# Set up the LLM
llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))

class GraphState(TypedDict):
    classification: Optional[str] = None         #To check who should speak next
    history: Optional[str] = None                #Conversation History
    current_response: Optional[str] = None       #Last line spoken by an agent
    count: Optional[int]=None                    #Conversation Length 
    results: Optional[str]=None                  #Result   
    greeting: Optional[str]=None                 #Welcome Message
    debate_topic: Optional[str] = None           #Topic for debate
    debater1: Optional[str] = None               #First Debater
    debater2: Optional[str] = None               #Second Debater

workflow = StateGraph(GraphState)

def build_prefix(debater1, debater2, topic, history, current_debater):
    previous_debater = ""

    if current_debater == debater1:
        previous_debater = debater2
    else:
        previous_debater = debater1

    prefix = f'''You are impersonating {debater1}  in a debate with {debater2} (Opp) on the topic: "{topic}".
    The debate so far is as follows:

    {history}

    Be sure to start the debate with an ad hominem attack on the opponent.
    If history is present, respond to {previous_debater}'s arguments, taking into account their arguments and beliefs. 
    Be sure to escalate your arguments as the debate progresses. 
    Respond  in a manner consistent with {current_debater}'s style and argumentation. 
    Provide a concise, impactful response that avoids repeating previous points.'''
    
    return prefix

# Function to classify sentiment of the input and decide who spoke it.
def classify(question, debater1, debater2):
    return llm(f"classify the sentiment of input as {debater1.replace(' ', '_')} or {debater2.replace(' ', '_')}. Output just the class. Input: {question}").strip()

# Function that generates the initial greeting message introducing the debaters and topic
def handle_greeting_node(state):
    return {"greeting": f"Welcome! Today's debate is between {state['debater1']} (Pro) and {state['debater2']} (Opp) on the topic of \"{state['debate_topic']}\"."}

def classify_input_node(state):
    question = state.get('current_response')
    classification = classify(question, state['debater1'], state['debater2'])
    return {"classification": classification}

def handle_pro(state):
    summary = state.get('history', '').strip()
    current_response = state.get('current_response', '').strip()
    prompt = build_prefix(state['debater1'], state['debater2'], state['debate_topic'], summary, state['debater1'])
    argument = f"{state['debater1']} : " + llm(prompt)
    return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}

def handle_opp(state):
    summary = state.get('history', '').strip()
    current_response = state.get('current_response', '').strip()
    prompt = build_prefix(state['debater1'], state['debater2'], state['debate_topic'], summary, state['debater2'])
    argument = f"{state['debater2']} : " + llm(prompt)
    return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count', 0) + 1}

def result(state):
    summary = state.get('history').strip()
    prompt = f"Summarize the conversation and judge who won the debate. No ties are allowed. Conversation: {summary}"
    return {"results": llm(prompt)}


# Define workflow nodes and edges
workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_pro", handle_pro)
workflow.add_node("handle_opp", handle_opp)
workflow.add_node("result", result)

def decide_next_node(state):
    # Alternate turns based on the count
    if state.get('count', 0) % 2 == 0:
        return "handle_pro"
    else:
        return "handle_opp"

def check_conv_length(state):
    return "result" if state.get("count", 0) == 4 else "classify_input"

workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_pro": "handle_pro",
        "handle_opp": "handle_opp"
    }
)

workflow.add_conditional_edges(
    "handle_pro",
    check_conv_length,
    {
        "result": "result",
        "classify_input": "classify_input"
    }
)

workflow.add_conditional_edges(
    "handle_opp",
    check_conv_length,
    {
        "result": "result",
        "classify_input": "classify_input"
    }
)

workflow.set_entry_point("handle_greeting")
workflow.add_edge('handle_greeting', "handle_pro")
workflow.add_edge('result', END)

# Compile the workflow

compiled_workflow = workflow.compile()

# Import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uvicorn import run as app_run

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define a Pydantic model for the request body
class DebateRequest(BaseModel):
    count: int = 0
    history: str = "Nothing"
    current_response: str = ""
    debate_topic: str = ""
    debater1: str = ""
    debater2: str = ""
    result: str = ""

@app.get("/", response_class=FileResponse)
async def read_index():
    return "index.html"  # Serving the index.html file directly

# Define an API endpoint
@app.post('/trigger_workflow')
async def trigger_workflow(request: DebateRequest):
    # Execute the workflow with the provided state
    conversation = compiled_workflow.invoke({
        'count': request.count,
        'history': request.history,
        'current_response': request.current_response,
        'debate_topic': request.debate_topic,
        'debater1': request.debater1,
        'debater2': request.debater2,
        'result': request.result
    })

    print(conversation['results'])

    # Return the history as a JSON response
    return {'history': conversation['history'], 'result': conversation['results']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)