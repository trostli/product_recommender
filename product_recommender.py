import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from datasets import load_dataset, DatasetDict

class ProductRecommender:
    def __init__(self):
        # Load the environment variables from the .env file
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Load data into LangChain
        dataset = load_dataset("LoganKells/amazon_product_reviews_video_games", trust_remote_code=True)
        dataset_dict = dataset

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir,"amazon_product_reviews_video_games.csv")
        dataset_dict["train"].to_csv(file_path)

        # Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        loader = CSVLoader(file_path=file_path)
        data = loader.load()
        chunked_documents = text_splitter.split_documents(data)

        # Create embedder
        store = LocalFileStore("./cache/")
        underlying_embeddings = OpenAIEmbeddings()
        embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )

        # Create vector store using FAISS
        vector_store = FAISS.from_documents(chunked_documents, embedder)
        vector_store.save_local("vector_store")

        # self.prompt = self.prompt_template.format(documents=formatted_documents, question=query)
        template = """Answer the question based only on the following context:\
            {context}
            Question: {question}
            """
        prompt = ChatPromptTemplate.from_template(template)

        retriever = vector_store.as_retriever()

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

        self.runnable_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def recommend(self, user_input):
        output_chunks = self.runnable_chain.invoke(user_input)
        print(output_chunks)
        return output_chunks
