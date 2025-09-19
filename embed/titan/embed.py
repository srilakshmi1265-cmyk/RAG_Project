import base64
import io
import json
import logging

import boto3
from langchain_aws import BedrockEmbeddings
from PIL import Image

# Configure logging with timestamp, log level, and message format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TitanTextEmbeddings:
    """
    A class for handling text embeddings using Amazon Titan embeddings via LangChain AWS integration.
    This class provides methods to embed text documents and queries.
    Requires AWS credentials and Bedrock access.
    """

    def __init__(self, aws_access_key_id: str = None, aws_secret_access_key: str = None, profile_name: str = None,
                 region_name: str = "us-east-1", model_id: str = "amazon.titan-embed-text-v1"):
        """
        Initialize the TitanTextEmbeddings.

        :param aws_access_key_id: AWS access key ID (optional if profile_name or default credentials are used).
        :param aws_secret_access_key: AWS secret access key (optional if profile_name or default credentials are used).
        :param profile_name: AWS profile name from credentials file (optional).
        :param region_name: AWS region (default: "us-east-1").
        :param model_id: ID of the Titan embedding model (default: "amazon.titan-embed-text-v1").
        """
        # Log initialization start
        logger.info(f"Initializing TitanTextEmbeddings with model_id: {model_id}, region: {region_name}")

        # Initialize BedrockEmbeddings based on provided credentials
        try:
            if aws_access_key_id and aws_secret_access_key:
                logger.debug("Using explicit AWS credentials")
                self.embeddings = BedrockEmbeddings(
                    model_id=model_id,
                    region_name=region_name,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                )
            elif profile_name:
                logger.debug(f"Using AWS profile: {profile_name}")
                self.embeddings = BedrockEmbeddings(
                    credentials_profile_name=profile_name,
                    model_id=model_id,
                    region_name=region_name,
                )
            else:
                logger.debug("Using default AWS credentials")
                self.embeddings = BedrockEmbeddings(
                    model_id=model_id,
                    region_name=region_name,
                )
            logger.debug("BedrockEmbeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BedrockEmbeddings: {e}")
            raise

        # Set the embedding dimension (fixed for Titan text embeddings)
        self.dimension = 1024  # Typical dimension for Titan text embeddings
        logger.debug(f"Set embedding dimension to {self.dimension}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text documents.

        :param texts: List of text strings to embed.
        :return: List of embedding vectors.
        """
        # Log the start of document embedding
        logger.info(f"Embedding {len(texts)} text documents")

        # Generate embeddings for the provided texts
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Generated {len(embeddings)} text document embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed text documents: {e}")
            raise

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single text query.

        :param query: Text query to embed.
        :return: Embedding vector.
        """
        # Log the start of query embedding
        logger.info("Embedding text query")

        # Generate embedding for the query
        try:
            embedding = self.embeddings.embed_query(query)
            logger.debug("Successfully embedded text query")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text query: {e}")
            raise


class TitanTextImageEmbeddings:
    """
    A class for handling text and image embeddings using Amazon Titan embeddings via LangChain AWS integration.
    This class provides methods to embed text documents, queries, and images.
    Requires AWS credentials and Bedrock access.
    """

    def __init__(self, aws_access_key_id: str = None, aws_secret_access_key: str = None, profile_name: str = None,
                 region_name: str = "us-east-1", model_id: str = "amazon.titan-embed-image-v1"):
        """
        Initialize the TitanTextImageEmbeddings.

        :param aws_access_key_id: AWS access key ID (optional if profile_name or default credentials are used).
        :param aws_secret_access_key: AWS secret access key (optional if profile_name or default credentials are used).
        :param profile_name: AWS profile name from credentials file (optional).
        :param region_name: AWS region (default: "us-east-1").
        :param model_id: ID of the Titan embedding model (default: "amazon.titan-embed-image-v1").
        """
        # Log initialization start
        logger.info(f"Initializing TitanTextImageEmbeddings with model_id: {model_id}, region: {region_name}")

        # Initialize BedrockEmbeddings for text embeddings
        try:
            if aws_access_key_id and aws_secret_access_key:
                logger.debug("Using explicit AWS credentials")
                self.embeddings = BedrockEmbeddings(
                    model_id=model_id,
                    region_name=region_name,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                )
                # Initialize boto3 Bedrock runtime client for image embeddings
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=region_name,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                )
            elif profile_name:
                logger.debug(f"Using AWS profile: {profile_name}")
                self.embeddings = BedrockEmbeddings(
                    credentials_profile_name=profile_name,
                    model_id=model_id,
                    region_name=region_name,
                )
                # Initialize boto3 Bedrock runtime client with profile
                session = boto3.Session(profile_name=profile_name)
                self.bedrock_client = session.client('bedrock-runtime', region_name=region_name)
            else:
                logger.debug("Using default AWS credentials")
                self.embeddings = BedrockEmbeddings(
                    model_id=model_id,
                    region_name=region_name,
                )
                # Initialize boto3 Bedrock runtime client with default credentials
                self.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
            logger.debug("BedrockEmbeddings and Bedrock runtime client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BedrockEmbeddings or Bedrock runtime client: {e}")
            raise

        # Set the embedding dimension (fixed for Titan image embeddings)
        self.dimension = 1024  # Typical dimension for Titan image embeddings
        logger.debug(f"Set embedding dimension to {self.dimension}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text documents.

        :param texts: List of text strings to embed.
        :return: List of embedding vectors.
        """
        # Log the start of document embedding
        logger.info(f"Embedding {len(texts)} text documents")

        # Generate embeddings for the provided texts
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Generated {len(embeddings)} text document embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed text documents: {e}")
            raise

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single text query.

        :param query: Text query to embed.
        :return: Embedding vector.
        """
        # Log the start of query embedding
        logger.info("Embedding text query")

        # Generate embedding for the query
        try:
            embedding = self.embeddings.embed_query(query)
            logger.debug("Successfully embedded text query")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text query: {e}")
            raise

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        """
        Embed a list of images from file paths using the AWS Bedrock runtime client.

        :param image_paths: List of image file paths.
        :return: List of embedding vectors.
        """
        # Log the start of image embedding
        logger.info(f"Embedding {len(image_paths)} images")

        # Load images from file paths
        images = []
        for path in image_paths:
            logger.debug(f"Loading image: {path}")
            try:
                images.append(Image.open(path))
                logger.debug(f"Successfully loaded image: {path}")
            except Exception as e:
                logger.error(f"Failed to load image {path}: {e}")
                raise

        # Generate embeddings for the images using Bedrock runtime client
        embeddings = []
        for i, image in enumerate(images):
            logger.debug(f"Generating embedding for image {i+1}/{len(images)}")
            try:
                # Convert image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Prepare request for Bedrock runtime
                request_body = {
                    "inputImage": img_base64
                }

                # Invoke the Titan model
                response = self.bedrock_client.invoke_model(
                    modelId="amazon.titan-embed-image-v1",
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json"
                )

                # Parse response
                response_body = json.loads(response['body'].read())
                embedding = response_body.get('embedding')
                if not embedding:
                    raise ValueError("No embedding returned from Bedrock API")
                embeddings.append(embedding)
                logger.debug(f"Successfully generated embedding for image {i+1}")
            except Exception as e:
                logger.error(f"Failed to embed image {i+1} ({image_paths[i]}): {e}")
                raise

        logger.debug(f"Generated {len(embeddings)} image embeddings")
        return embeddings