from typing import List
import os
import logging
import os
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from embed.titan.embed import TitanTextEmbeddings, TitanTextImageEmbeddings

vectorstore = OpenSearchVectorSearch(
    index_name="ragtitan",
    embedding_function= TitanTextImageEmbeddings(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    ),  # e.g., OpenAIEmbeddings()
    opensearch_url="https://search-clinicaldatadomain-xqpcko5eaq5dnxxc4okbwpj2uy.aos.us-east-1.on.aws/",
    http_auth=("Self_Admin123", "Srilu@123"),  # master user creds
)

class OpenSearchRetriever(BaseRetriever):
    vectorstore: OpenSearchVectorSearch
    top_k: int = 4
    min_score: float = 0.0

    """
    A retriever class that extends BaseRetriever for querying an OpenSearch vectorstore.
    This class uses the provided embeddings to embed queries and retrieve relevant documents.
    """

# def __init__(
#          self,
#          vectorstore: OpenSearchVectorSearch,
#          top_k: int = 4,
#          min_score: float = 0.0,
#       ):
    """
        Initialize the OpenSearchRetriever.

        :param vectorstore: The LangChain OpenSearchVectorSearch instance.
        :param top_k: Number of top results to return (default: 4).
        :param min_score: Minimum similarity score threshold (default: 0.0).
    """
        # super().__init__()
        # self.vectorstore = vectorstore
        # self.top_k = top_k
        # self.min_score = min_score

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Retrieve relevant documents for a given query.

        :param query: The search query.
        :param run_manager: Callback manager for the retriever run.
        :return: List of retrieved Document objects.
        """
        # Perform similarity search with score
        results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        # Filter by min_score (note: scores in OpenSearch are typically cosine similarity, range -1 to 1)
        docs = [doc for doc, score in results if score >= self.min_score]
        return docs
if __name__ == "__main__":
    retriever = OpenSearchRetriever(
        vectorstore=vectorstore,
        top_k=3,
        min_score=0.3
    )
    query = "what is BLEU value for GNMT?"
    docs = retriever.get_relevant_documents(query)
    for d in docs:
        print(f"Doc: {d.page_content[:200]}...\n")
        print("Documents retreived")