import logging
import os

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

from langchain import hub

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever

from langchain.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set your Azure OpenAI API credentials
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-04-01-preview"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4-32k"

# Initialize the LLM and embeddings
llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    model_version="0613",
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15",
)

# Load the FAISS index
db = FAISS.load_local("C:/Users/LAWJusHM/Source/Repos/ChatGPTFlask/vdb.index", embeddings, allow_dangerous_deserialization=True)

# Create the retrieval chain
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

def get_response(msg):
    try:
        response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=msg), response["answer"]])
        content = response["answer"]
        return content
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return "An error occurred. Please try again later."