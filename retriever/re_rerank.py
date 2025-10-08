from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank  # Updated import
from langchain_core.retrievers import BaseRetriever
from pydantic import SecretStr
class RerankedRetriever(ContextualCompressionRetriever):
    def __init__(
            self,
            base_retriever: BaseRetriever,
            cohere_api_key: str,
            top_n: int = 3,
            model: str = "rerank-english-v3.0",
    ):
        """
        Initialize the RerankedRetriever.

        :param base_retriever: The base retriever instance (e.g., PineconeRetriever or OpenSearchRetriever).
        :param cohere_api_key: API key for Cohere reranking service.
        :param top_n: Number of top documents to return after reranking (default: 3).
        :param model: Cohere reranking model to use (default: "rerank-english-v3.0").
        """
        compressor = CohereRerank(
            cohere_api_key=SecretStr(cohere_api_key),
            top_n=top_n,
            model=model
        )
        super().__init__(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )