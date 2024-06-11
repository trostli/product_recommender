import chainlit as cl
from chainlit import run_sync
from crewai import Agent, Task, Crew
from crewai_tools import tool

from product_recommender import ProductRecommender

recommender = ProductRecommender()

@tool("Ask Human follow up questions")
def ask_human(question: str) -> str:
    """Ask human follow up questions"""
    human_response  = run_sync( cl.AskUserMessage(content=f"{question}").send())
    if human_response:
        return human_response["output"]

@tool("Tool Name")
def get_recommendations(question: str) -> str:
    """Takes a summary of the users video game interests and returns a list of games that match the users interests."""
    # Tool logic here
    recommendation_prompt = f"""
    You are a video game curator and recommender.
    Your expertise lies in a wide swath of video games and their reviews.
    Your job is to take a summary of the user's interest and use the product_recommender to get a list of games that match the user's interests.
    In your answer mention what some of the reviewers thought of the game and why.
    Please recommend a game for the following user profile: {question}
    """
    recommendation = recommender.recommend(recommendation_prompt)
    return recommendation

@cl.on_chat_start
def on_chat_start():
    print("A new chat session has started!")
    interviewer = Agent(
        role='Video game interest interviewer',
        goal='Learn about the user and summarize their video gameinterests',
        backstory="""You are a video game curator and recommender.
            Your expertise lies in a wide swath of video games and their reviews.
            In order to make recommendations you need to learn about the user.
            To learn about the user you need to ask follow up questions.
            You can ask them about their interests, their play style, their preferred genre, etc.
            You can also ask them about their play time, their preferred platform, etc.
            When you have enough information to make a recommendation you can creeate a summary of the user's video game interests.
            """,
            # verbose=True,
            allow_delegation=False,
            tools= [ask_human]
        )

    recommender = Agent(
        role='Video game recommender',
        goal='Recommend a list of video games based on a user summary',
        backstory="""You are a video game curator and recommender.
            Your expertise lies in a wide swath of video games and their reviews.
            Your job is to take a summary of the user's interest and use the product_recommender to get a list of games that match the user's interests.
            """,
            # verbose=True,
            allow_delegation=False,
            tools= [get_recommendations]
        )

    extraction = Task(
        description="""Extract key information about the user produce a summary of their video game interests""",
        expected_output="A summary of the users video game interests",
        agent=interviewer
    )

    recommendation = Task(
        description="""Uses a users summary of their video game interests to recommend a list of games""",
        expected_output="The recommended games in list format",
        agent=recommender,
        callback=lambda output: run_sync(cl.Message(content=output.raw_output).send())
    )

    crew = Crew(
        agents=[interviewer, recommender],
        tasks=[extraction, recommendation],
        process="sequential",
        verbose=2, # You can set it to 1 or 2 to different logging levels
    )
    crew.kickoff()

# @cl.on_message
# async def main(message: cl.Message):
#     recommendation = recommender.recommend(message.content)
#     await cl.Message(content=recommendation).send()

@tool("Ask Human follow up questions")
def ask_human(question: str) -> str:
    """Ask human follow up questions"""
    human_response  = run_sync( cl.AskUserMessage(content=f"{question}").send())
    if human_response:
        return human_response["output"]