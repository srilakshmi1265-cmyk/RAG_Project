# Purpose:
# This is a production-ready RAG pipeline API that connects OpenSearch, AWS Bedrock LLM (Claude), and optionally Cohere reranking — all exposed via FastAPI endpoints.
# Output:
# Query → Retrieve docs → Format → Prompt → Bedrock LLM → Answer
#This is the file used for retreive image in docker
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import boto3
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from embed.titan.embed import TitanTextEmbeddings, TitanTextImageEmbeddings
from generator.ge_aws_bedrock import BedrockGenerator
from retriever.re_opensearch import OpenSearchRetriever
from retriever.re_rerank import RerankedRetriever

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

app = FastAPI(title='RAG-API')

embeddings = TitanTextImageEmbeddings()
vectorstore_for_OSVS = OpenSearchVectorSearch(
    opensearch_url="https://search-clinicaldatadomain-xqpcko5eaq5dnxxc4okbwpj2uy.aos.us-east-1.on.aws/",
    index_name='ragtitan',
    embedding_function=embeddings,
    http_auth=("Self_Admin123", "Srilu@123")
)
os_retriever = OpenSearchRetriever(
    vectorstore=vectorstore_for_OSVS,
    top_k=4
)
# Try Cohere reranker if key exists

COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")

if COHERE_API_KEY:
    print("✅ Using Cohere reranker for better ranking")

reranked_retriever = RerankedRetriever(os_retriever, cohere_api_key=COHERE_API_KEY)


def format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join([
        f"Document {i + 1}:\n{doc.page_content}\n"
        f"Metadata: {doc.metadata}"
        for i, doc in enumerate(docs)
    ])

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the question. If you don't know the answer 
based on the context, just say that you don't know.

Context:
{context}

Question: {question}

Answer: Provide a comprehensive and accurate answer based on the context above.
""")

# bedrock = BedrockGenerator(
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
#     )

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  # change if needed
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)



llm = BedrockChat(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    client=bedrock_client,
)

rag_chain = (

        {
            'question': RunnablePassthrough(),
            'context': reranked_retriever | format_docs

        }
        | rag_prompt
        | llm
        | StrOutputParser()

)


# Alternative: More detailed RAG chain with intermediate steps
class DetailedRAGChain:
    def __init__(self, retriever, llm, prompt):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt


def invoke(self, query: str):
    # Step 1: Retrieve relevant documents
    retrieved_docs = self.retriever.invoke(query)

    # Step 2: Format context
    context = format_docs(retrieved_docs)

    # Step 3: Generate prompt
    formatted_prompt = self.prompt.format(context=context, question=query)

    # Step 4: Generate response
    response = self.llm.invoke(formatted_prompt)

    return {
        "retrieved_docs": retrieved_docs,
        "context": context,
        "response": response.content,
        "metadata": {
            "num_docs_retrieved": len(retrieved_docs),
            "retrieval_scores": [getattr(doc.metadata, 'score', None) for doc in retrieved_docs]
        }
    }


# Initialize detailed RAG chain
detailed_rag_chain = DetailedRAGChain(reranked_retriever, llm, rag_prompt)


# Pydantic models
class QueryRequest(BaseModel):
    query: str


use_detailed_chain: bool = False


class QueryResponse(BaseModel):
    context: List[dict]


llm_output: str
metadata: dict = None


class SimpleRAGResponse(BaseModel):
    answer: str


@app.post("/rag", response_model=SimpleRAGResponse)
async def simple_rag(request: QueryRequest):
    """
       Simple RAG endpoint using LangChain's chain composition.
       Returns just the generated answer.
   """

    try:
       answer = rag_chain.invoke(request.query)
       return {"answer": answer}
    except Exception as e:
       raise HTTPException(status_code=400, detail=str(e))

@app.post("/rag-detailed", response_model=QueryResponse)
async def detailed_rag(request: QueryRequest):
    """
   Detailed RAG endpoint that returns intermediate steps and metadata.
   Useful for debugging and understanding the RAG process.
   """
    try:
       if request.use_detailed_chain:
           result = detailed_rag_chain.invoke(request.query)
           return {
            "context": [doc.dict() for doc in result["retrieved_docs"]],
            "llm_output": result["response"],
            "metadata": result["metadata"]
                  }
       else:
          # Use simple chain but return docs separately
        retrieved_docs = reranked_retriever.invoke(request.query)
        answer = rag_chain.invoke(request.query)
        return {
            "context": [doc.dict() for doc in retrieved_docs],
            "llm_output": answer,
            "metadata": {"num_docs_retrieved": len(retrieved_docs)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Keep your original endpoint for backward compatibility
@app.post("/retrieve", response_model=QueryResponse)
async def retrieve_documents(request: QueryRequest):
    """
   Original endpoint for backward compatibility.
   """


    try:
       retrieved_docs = reranked_retriever.invoke(request.query)
       answer = rag_chain.invoke(request.query)
       return {
    "context": [doc.dict() for doc in retrieved_docs],
    "llm_output": answer
       }

    except Exception as e:
     raise HTTPException(status_code=500, detail=str(e))
     print(f"❌ Error during retrieval or generation: {e}")
# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RAG Chain API is running"}
if __name__ == '__main__':
                         import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
