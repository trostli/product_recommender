import chainlit as cl

from product_recommender import ProductRecommender

recommender = ProductRecommender()

@cl.on_message
async def main(message: cl.Message):
    recommendation = recommender.recommend(message.content)
    await cl.Message(content=recommendation).send()