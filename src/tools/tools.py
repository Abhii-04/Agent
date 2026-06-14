from playwright.async_api import Browser, async_playwright, async_playwright
from dotenv import load_dotenv
import os 
import requests
from langchain_core.tools import Tool
from langchain_community.agent_toolkits import FileManagementToolkit,PlayWrightBrowserToolkit
from langchain_experimental.tools import PythonAstREPLTool, PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)



async def playwright():
    """ use this tool for browser automatin"""
    playwright=await async_playwright().start()
    browser=await playwright.chromium.launch(
        channel='chrome',
        headless=False
    )
    toolkit=PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(),browser,playwright

def get_file():
    """ use this tool for file managment"""
    toolkit=FileManagementToolkit(root_dir='sandbox')
    return toolkit.get_tools()

def serper():
    """Use this tool to searh google for latest information """
    serper=GoogleSerperAPIWrapper()
    return Tool(
        name='google-search',
        func=serper.run,
        description="search Google for latest information"
    )

def python_repl():
    """ use this tool for file managment"""
    python_repl=PythonREPLTool()
    return python_repl


def mcp():
    client=MultiServerMCPClient(
        {}
    )

    return