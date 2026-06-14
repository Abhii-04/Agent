from concurrent.futures.process import _system_limited
from email import message
from src.prompt import system_prompt,feedback_prompt,evaluator_prompt
from playwright.async_api import Browser
from src.tools.tools import playwright,get_file,serper,python_repl
from dotenv import load_dotenv
import os
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from typing import Annotated,List,Any,Optional,Dict,TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import uuid
from pathlib import Path
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
import asyncio
from langgraph.prebuilt import ToolNode




load_dotenv(override=True)

llm=ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    model='deepseek-v4-flash',
    base_url='https://api.deepseek.com'
)

class State(TypedDict):
    messages:Annotated[List[Any],add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met:bool
    user_input_needed: bool
    evaluation_count: int

class EvaluatorOutput(BaseModel):
    feedback:str=Field(description="Feednacl on assistants response")
    success_criteria_met:bool=Field(description="whether the success criteria have been met")
    
class Agent:
    def __init__(self):
        self.worker_agent=None
        self.worker_agent_with_tools=None
        self.evaluator_llm_with_output = None
        self.tools=None
        self.graph=None
        self.agent_id=str(uuid.uuid4())
        self.browser=None
        self.playwright=None


    async def setup(self):
        self.playwright_tools,self.browser,self.playwright=await playwright()

        self.file_tool=get_file()
        self.serper_tool=serper()
        self.python_repl_tool=python_repl()

        self.tools=(
            self.playwright_tools
            + self.file_tool
            +  [self.serper_tool]
            + [self.python_repl_tool]

        )
        worker_agent=llm
        self.worker_agent_with_tools= worker_agent.bind_tools(self.tools)
        evaluator_agent=llm
        self.evaluator_llm_with_output= evaluator_agent

        await self.build_graph()

    def worker(self, state:State)->Dict[str,Any]:
        system_message=system_prompt

        if state.get(f"feedback_on_work"):
            system_message += feedback_prompt

        found_system_message=False
        messages= state["messages"]

        for message in messages:
            if isinstance(message,SystemMessage):
                message.content=system_message
                found_system_message=True

        if not found_system_message:
            messages= [SystemMessage(content=system_message)]+messages

        response=self.worker_agent_with_tools.invoke(messages)

        return {
            "messages": [response]
        }

    def worker_router(self, state:State)->str:
        last_message=state["messages"][-1]

        if hasattr(last_message,"tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"
    
    def format_conversation(self,messages:List[Any]) ->str:
        conversation="conversation history:\n\n"
        for message in messages:
            if isinstance(message,HumanMessage):
               conversation += f"user: {message.content}\n"

            elif isinstance(message,AIMessage):
                text=message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation
    

    def evaluator(self,state:State)-> State:
        last_response=state["messages"][-1].content

        system_message=evaluator_prompt

        user_message= f"""The conversation is:

{self.format_conversation(state["messages"])}

Success criteria:
{state["success_criteria"]}

Assistant final response:
{last_response}
"""

        evaluator_message = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        eval_result=self.evaluator_llm_with_output.invoke(
            evaluator_message
        )
        content=eval_result.content
        lower_content=content.lower()
        success_criteria_met= (
            "success: true" in lower_content
            or "success_criteria_met: true" in lower_content
            or "success criteria met: true" in lower_content
        )
        user_input_needed=(
            "user_input_needed: true" in lower_content
            or "user input needed: true" in lower_content
        )
        new_state={
            "messages": [
                AIMessage(content=f"evaluator feedback on this answer: {content}")
            ],
            "feedback_on_work": content,
            "success_criteria_met": success_criteria_met,
            "user_input_needed": user_input_needed,
            "evaluation_count": state.get("evaluation_count", 0) + 1,
        }
        return new_state

    
    def route_based_on_evaluation(self,state:State) ->str:
        if (
            state["success_criteria_met"]
            or state["user_input_needed"]
            or state.get("evaluation_count", 0) >= 1
        ):
            return "END"
        else:
            return "worker"
    
    async def build_graph(self):
        graph_builder=StateGraph(State)

        graph_builder.add_node("worker",self.worker)
        graph_builder.add_node("evaluator",self.evaluator)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))

        graph_builder.add_conditional_edges(
            "worker",
            self.worker_router,
            {"tools":"tools","evaluator":"evaluator"},
        )

        graph_builder.add_edge("tools","worker")
        graph_builder.add_conditional_edges(
            "evaluator",
            self.route_based_on_evaluation,
            {"worker":"worker","END":END},
        )
        graph_builder.add_edge(START,"worker")

        self.graph=graph_builder.compile()

    async def run_superstep(self,message,success_criteria,history):
        config = {"configurable": {"thread_id": self.agent_id}}

        state = {
            "messages": [HumanMessage(content=message)],
            "success_criteria": success_criteria
            or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "evaluation_count": 0,
        }
        result = await self.graph.ainvoke(state, config=config)

        user = {"role": "user", "content": message}

        assistant_messages = [
            msg for msg in result["messages"]
            if isinstance(msg, AIMessage)
            and msg.content
            and not msg.content.startswith("evaluator feedback on this answer:")
        ]
        feedback_messages = [
            msg for msg in result["messages"]
            if isinstance(msg, AIMessage)
            and msg.content
            and msg.content.startswith("evaluator feedback on this answer:")
        ]

        reply = {
            "role": "assistant",
            "content": assistant_messages[-1].content if assistant_messages else "No response generated.",
        }

        feedback = {
            "role": "assistant",
            "content": feedback_messages[-1].content if feedback_messages else "No evaluator feedback generated.",
        }

        return history + [user, reply, feedback]
    async def cleanup_async(self):
        if self.browser:
            await self.browser.close()
            self.browser = None

        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

    def cleanup(self):
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(self.cleanup_async())

        except RuntimeError:
            asyncio.run(self.cleanup_async())
            return None
