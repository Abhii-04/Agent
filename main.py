import os
from dotenv import load_dotenv
import streamlit as st
from src.agent import Agent
import asyncio

load_dotenv(override=True)

async def setup():
    agent = Agent()
    await agent.setup()
    return agent

async def process_message(agent,message,success_criteria,history):
    results=await agent.run_superstep(message,success_criteria,history)

    return results,agent

async def reset():
    new_agent=Agent()
    await new_agent.setup()
    return "","",None,new_agent


def free_resources(agent):
    print("cleaning up")
    try:
        if agent:
            run_async(agent.cleanup_async())
    except Exception as e:
        print(f"Exception during cleanup: {e}")

def run_async(coro):
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()

    return st.session_state.event_loop.run_until_complete(coro)


def initialize_session():
    if "agent" not in st.session_state:
        with st.spinner("Starting agent..."):
            st.session_state.agent = run_async(setup())

    if "history" not in st.session_state:
        st.session_state.history = []

    if "success_criteria" not in st.session_state:
        st.session_state.success_criteria = "The answer should be clear and accurate"


st.set_page_config(page_title="Agent", layout="wide")
st.title("Agent")

initialize_session()

with st.sidebar:
    st.header("Settings")
    st.session_state.success_criteria = st.text_area(
        "Success criteria",
        value=st.session_state.success_criteria,
        height=120,
        help="Describe what a successful answer should satisfy.",
    )

    if st.button("Reset conversation", use_container_width=True):
        free_resources(st.session_state.get("agent"))
        with st.spinner("Resetting agent..."):
            _, _, _, st.session_state.agent = run_async(reset())
        st.session_state.history = []
        st.rerun()

for message in st.session_state.history:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask the agent...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                st.session_state.history, st.session_state.agent = run_async(
                    process_message(
                        st.session_state.agent,
                        prompt,
                        st.session_state.success_criteria,
                        st.session_state.history,
                    )
                )
                st.markdown(st.session_state.history[-2]["content"])

                with st.expander("Evaluator feedback"):
                    st.markdown(st.session_state.history[-1]["content"])
            except Exception as e:
                st.error(f"Agent failed: {e}")
