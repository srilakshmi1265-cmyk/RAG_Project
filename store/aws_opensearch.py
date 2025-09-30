#This script extracts text and images from a PDF, generates embeddings using AWS Titan, and then stores both text and image vectors in AWS OpenSearch Service for semantic search
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from embed.clip.embed import CLIPEmbeddings
from embed.titan.embed import TitanTextEmbeddings, TitanTextImageEmbeddings
from extract.LLamaParse.extract_page import LlamaParseLoader
import logging
import boto3
from store.base import StoreBase
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from opensearchpy import OpenSearch

# Configure logging with timestamp, log level, and message format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenSearchStore(StoreBase):
    """
    A class for storing embedded documents into OpenSearch, extending StoreBase.
    Uses LangChain's OpenSearchVectorSearch for integration.
    """

    def store(
        self,
        opensearch_url: str,
        index_name: str,
        http_auth: tuple = None,  # (username, password)
        use_ssl: bool = True,
        verify_certs: bool = True,
    ) -> OpenSearchVectorSearch:
        """
        Store the embedded documents into OpenSearch.

        :param opensearch_url: URL for the OpenSearch instance (e.g., "https://localhost:9200").
        :param index_name: Name of the OpenSearch index to use or create.
        :param http_auth: Optional HTTP authentication tuple (username, password).
        :param use_ssl: Use SSL for connection (default: True).
        :param verify_certs: Verify SSL certificates (default: True).
        :return: The LangChain OpenSearchVectorSearch instance.
        """
        # Log the start of the store operation
        logger.info(f"Starting store operation for OpenSearch index: {index_name}, URL: {opensearch_url}")

        # Initialize OpenSearch client
        logger.debug("Initializing OpenSearch client")
        try:
            client = OpenSearch(
                hosts=[opensearch_url],
                http_auth=http_auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
            )
            logger.debug("OpenSearch client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            raise

        # Create vectorstore for text documents using LangChain's OpenSearchVectorSearch
        logger.debug(f"Creating vectorstore for {len(self.docs)} text documents")
        try:
            vectorstore = OpenSearchVectorSearch.from_documents(
                documents=self.docs,
                embedding=self.embeddings,  # Use embed_query as the embedding function
                opensearch_url=opensearch_url,
                index_name=index_name,
                http_auth=http_auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
            )
            logger.info(f"Vectorstore created for index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to create vectorstore for index {index_name}: {e}")
            raise

        # Prepare and upsert image vectors if available
        logger.debug("Preparing image upserts")
        image_upserts = self._prepare_image_upserts()
        if image_upserts:
            logger.info(f"Upserting {len(image_upserts)} image vectors to index: {index_name}")
            for i, upsert in enumerate(image_upserts):
                doc_id = upsert['id']
                vector = upsert['values']
                metadata = upsert['metadata']
                logger.debug(f"Upserting image {i+1}/{len(image_upserts)} with ID: {doc_id}")
                try:
                    # Upsert image vector to OpenSearch
                    client.index(
                        index=index_name,
                        id=doc_id,
                        body={
                            "vector_field": vector,  # Assuming 'vector_field' is the knn_vector field
                            **metadata
                        }
                    )
                    logger.debug(f"Successfully upserted image vector with ID: {doc_id}")
                except Exception as e:
                    logger.error(f"Failed to upsert image vector with ID {doc_id}: {e}")
                    raise
        else:
            logger.debug("No image vectors to upsert")

        # Log completion of the store operation
        logger.info(f"Store operation completed for index: {index_name}")
        return vectorstore
if __name__ == "__main__":
    loader = LlamaParseLoader(
        "C:/Users/own/Downloads/Learn-RAG-code-only/Learn-RAG-code-only/pdf_files/transformer.pdf",
        "key",
        openai_api_key="open search key",
        describe_images=True,
        image_dir="C:/Users/own/Downloads/Learn-RAG-code-only 2_Updated_Code/Learn-RAG-code-only/extracted_images")

    #clip_embeddings = CLIPEmbeddings()
    titan_embeddings = TitanTextImageEmbeddings(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    # Step 4: Call store() to push to AWS OpenSearch
    # region = "us-east-1"
    # service = "es"
    # credentials = boto3.Session().get_credentials()
    vectorstore = OpenSearchStore(loader,titan_embeddings)
    vectorstore.store(
        opensearch_url="https://search-clinicaldatadomain-xqpcko5eaq5dnxxc4okbwpj2uy.aos.us-east-1.on.aws/",
        index_name="ragtitan",
        http_auth=("Self_Admin123", "Srilu@123"),  # if fine-grained access control enabled
        use_ssl=True,
        verify_certs=True
    )

    print("✅ Data indexed into OpenSearch!")
    # # List all indices
    # print(vectorstore.cat.indices())
    # #
    # # # Check if a specific index exists
    # if vectorstore.indices.exists("rag"):
    #    print("Index exists ✅")
    # else:
    #    print("Index not found ❌")