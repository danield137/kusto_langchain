import getpass
import os

import bs4
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from langchain_kusto.vectorstore import KustoVectorStore

from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from azure.identity import DefaultAzureCredential

import dotenv

dotenv.load_dotenv()

credential = DefaultAzureCredential(
    exclude_workload_identity_credential=True,
    exclude_shared_token_cache_credential=True,
    exclude_interactive_browser_credential=False)

os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

vector_store = KustoVectorStore(
    cluster_uri=os.environ["AZURE_KUSTO_CLUSTER_URI"],
    database=os.environ["AZURE_KUSTO_DATABASE"],
    collection_name=os.environ["AZURE_KUSTO_COLLECTION"],
    embedding=embeddings,
    embedding_column="embedding_text",
    id_column="vector_id",
    content_column="doc_text",
)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Test the application with a sample question
if __name__ == "__main__":
    # Example question - you can change this to test different queries
    test_question = "What city is mentioned in the Pokemon games?"
    
    print(f"Question: {test_question}")
    print("Retrieving relevant documents and generating answer...")
    
    # Run the graph with the test question
    result = graph.invoke({"question": test_question}) # type: ignore
    print(f"\nAnswer: {result['answer']}")
    print(f"\nRetrieved {len(result['context'])} relevant documents")
    
    # Optionally, you can add a loop to ask multiple questions
    # Uncomment the following code if you want interactive questioning:
    """
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = graph.invoke({"question": question})
        print(f"\nAnswer: {result['answer']}")
        print(f"Retrieved {len(result['context'])} relevant documents")
    """
