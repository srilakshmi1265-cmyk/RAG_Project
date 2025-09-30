
import uuid
import logging

# Configure logging with timestamp, log level, and message format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StoreBase:
    """
    Abstract base class for storing embedded documents into a vector database.
    Subclasses must implement the _store method to handle database-specific storage.
    """

    def __init__(self, loader, embeddings):
        """
        Initialize the StoreBase.

        :param loader: Instance of LlamaParseLoader to load and parse documents.
        :param embeddings: Instance of an embedding class (e.g., CLIPEmbeddings, ImageBindEmbeddings).
        """
        # Log initialization start
        logger.info("Initializing StoreBase")

        # Store the loader and embeddings instances for use in document processing
        self.loader = loader
        self.embeddings = embeddings

        # Load documents from the provided loader
        logger.debug("Loading documents from loader")
        try:
            self.docs = self.loader.load()
            logger.info(f"Successfully loaded {len(self.docs)} documents")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
        logger.debug("StoreBase initialization completed")

    def _prepare_text_upserts(self) -> list[dict]:
        """Helper method to prepare text upserts (vectors and metadata)."""
        # Log the start of text upsert preparation
        logger.debug("Preparing text upserts")

        # Initialize empty list for text upserts
        text_upserts = []

        # Extract page content from documents for embedding
        text_texts = [doc.page_content for doc in self.docs]
        logger.debug(f"Extracted {len(text_texts)} text documents for embedding")

        # Check if there are texts to embed
        if text_texts:
            # Generate embeddings for text documents
            logger.debug("Generating text embeddings")
            try:
                text_vectors = self.embeddings.embed_documents(text_texts)
                logger.debug(f"Generated {len(text_vectors)} text vectors")
            except Exception as e:
                logger.error(f"Failed to generate text embeddings: {e}")
                raise

            # Create upsert dictionaries with vector, metadata, and unique ID
            for i, (vector, doc) in enumerate(zip(text_vectors, self.docs)):
                metadata = {**doc.metadata, 'type': 'text', 'text': doc.page_content}
                text_upserts.append({
                    'id': str(uuid.uuid4()),
                    'values': vector,
                    'metadata': metadata
                })
                logger.debug(f"Prepared text upsert {i + 1}/{len(text_vectors)} with ID: {text_upserts[-1]['id']}")
        else:
            logger.debug("No text documents to prepare for upsert")

        # Log completion and return prepared upserts
        logger.info(f"Prepared {len(text_upserts)} text upserts")
        return text_upserts

    def _prepare_image_upserts(self) -> list[dict]:
        """Helper method to prepare image upserts (vectors and metadata) if supported."""
        # Log the start of image upsert preparation
        logger.debug("Preparing image upserts")

        # Initialize lists to collect image paths and metadata
        image_upserts = []
        all_image_paths = []
        image_metadatas = []

        # Iterate through documents to collect image paths and metadata
        for doc in self.docs:
            if 'images' in doc.metadata:
                images = doc.metadata['images']
                descriptions = doc.metadata.get('image_descriptions', ["" for _ in images])
                logger.debug(f"Found {len(images)} images in document page {doc.metadata['page']}")

                # Collect metadata for each image
                for idx, img_path in enumerate(images):
                    all_image_paths.append(img_path)
                    image_metadatas.append({
                        'type': 'image',
                        'image_path': img_path,
                        'source_page': doc.metadata['page'],
                        'text': f"Image from page {doc.metadata['page']}",  # Placeholder text
                        'description': descriptions[idx],
                    })
                    logger.debug(f"Collected metadata for image {idx + 1} on page {doc.metadata['page']}: {img_path}")

        # Check if there are images and if embeddings support image processing
        if all_image_paths and hasattr(self.embeddings, 'embed_images'):
            # Generate embeddings for images
            logger.debug(f"Generating embeddings for {len(all_image_paths)} images")
            try:
                image_vectors = self.embeddings.embed_images(all_image_paths)
                logger.debug(f"Generated {len(image_vectors)} image vectors")
            except Exception as e:
                logger.error(f"Failed to generate image embeddings: {e}")
                raise

            # Create upsert dictionaries for images with vector, metadata, and unique ID
            for vector, metadata in zip(image_vectors, image_metadatas):
                image_upserts.append({
                    'id': str(uuid.uuid4()),
                    'values': vector,
                    'metadata': metadata
                })
                logger.debug(f"Prepared image upsert for {metadata['image_path']} with ID: {image_upserts[-1]['id']}")
        else:
            logger.debug("No images or image embedding support available for upsert")

        # Log completion and return prepared upserts
        logger.info(f"Prepared {len(image_upserts)} image upserts")
        return image_upserts

    def store(self, **kwargs):
        """
        Abstract method to store the embedded documents into the database.
        Must be implemented by subclasses.

        :param kwargs: Database-specific parameters.
        :return: The database client or index instance.
        """
        # Log attempt to call abstract method
        logger.error("store method called on abstract StoreBase class")
        raise NotImplementedError("Subclasses must implement the store method.")