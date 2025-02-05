from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from src.qdrant.qdrant_utils import find_recipes


def create_agent(client):
    """
    Creates and initializes an agent using provided client and configuration.

    The function sets up an OpenAI GPT-4 based agent capable of finding recipes
    as per the user's ingredients by utilizing a tool named "Recipe Search".
    The agent is integrated with the OpenAI's system using the provided client
    and specific parameters.

    Usage of this function provides a fully initialized agent ready to perform
    task-specific actions with verbose logging for better debugging and insights.

    :param client: The client object required to interact with the OpenAI system.
                   This client acts as an intermediary to configure and manage
                   the agent's behavior.
    :return: The initialized agent object ready for execution of tasks.
    :rtype: Agent
    """

    def recipe_search_tool(user_ingredients):
        return find_recipes(client, user_ingredients)

    search_tool = Tool(
        name="Recipe Search",
        func=recipe_search_tool,
        description="Trova ricette in base agli ingredienti forniti dall'utente."
    )

    llm = ChatOpenAI(model="gpt-4")
    return initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
