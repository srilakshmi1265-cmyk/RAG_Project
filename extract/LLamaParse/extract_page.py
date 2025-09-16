import base64
import pathlib
import os
import logging

from langchain_core.documents import Document
from langchain.document_loaders.base import BaseLoader
from llama_cloud_services import LlamaParse
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaParseLoader(BaseLoader):
    """
    A custom LangChain loader for parsing PDF files using LlamaParse from LlamaCloud.
    This loader extracts text, markdown, images, charts, and structured data (e.g., tables)
    from PDFs and converts them into LangChain Document objects.
    Supports generating descriptions/summaries for extracted images and appending them
    below placeholders in the markdown content.
    """

    def __init__(
        self,
        pdf_path: str,
        api_key: str,
        image_dir: str,
        parse_mode: str = "parse_page_with_agent",
        model: str = "openai-gpt-4-1-mini",
        high_res_ocr: bool = True,
        adaptive_long_table: bool = True,
        outlined_table_extraction: bool = True,
        output_tables_as_html: bool = True,
        include_screenshot_images: bool = False,
        include_object_images: bool = True,
        verbose: bool = True,
        num_workers: int = 4,
        describe_images: bool = True,
        openai_api_key: str = None,
        vision_model: str = "gpt-4o",
    ):
        """
        Initialize the LlamaParseLoader.

        :param pdf_path: Path to the PDF file to parse.
        :param api_key: API key for LlamaCloud services.
        :param image_dir: Directory to save extracted images (default: "extracted_images/").
        :param parse_mode: Parsing mode for LlamaParse (default: "parse_page_with_agent").
        :param model: Model to use for parsing (default: "openai-gpt-4-1-mini").
        :param high_res_ocr: Enable high-resolution OCR (default: True).
        :param adaptive_long_table: Enable adaptive long table extraction (default: True).
        :param outlined_table_extraction: Enable outlined table extraction (default: True).
        :param output_tables_as_html: Output tables as HTML (default: True).
        :param include_screenshot_images: Include screenshot images in extraction (default: False).
        :param include_object_images: Include object images in extraction (default: True).
        :param verbose: Enable verbose output during parsing (default: True).
        :param num_workers: Number of workers for parallel processing (default: 4).
        :param describe_images: If True, generate descriptions for images and append below placeholders (default: True).
        :param openai_api_key: API key for OpenAI (required if describe_images=True).
        :param vision_model: OpenAI vision model for image description (default: "gpt-4o").
        """
        logger.info(f"Initializing LlamaParseLoader with pdf_path: {pdf_path}, image_dir: {image_dir}")
        self.pdf_path = pdf_path
        self.api_key = api_key
        self.image_dir = image_dir
        self.parse_mode = parse_mode
        self.model = model
        self.high_res_ocr = high_res_ocr
        self.adaptive_long_table = adaptive_long_table
        self.outlined_table_extraction = outlined_table_extraction
        self.output_tables_as_html = output_tables_as_html
        self.include_screenshot_images = include_screenshot_images
        self.include_object_images = include_object_images
        self.verbose = verbose
        self.num_workers = num_workers
        self.describe_images = describe_images
        # Create the image directory if it doesn't exist
        logger.debug(f"Creating image directory: {image_dir}")
        pathlib.Path(image_dir).mkdir(exist_ok=True)
        # Initialize OpenAI client if describing images
        if self.describe_images:
            if not openai_api_key:
                logger.error("OpenAI API key is missing when describe_images is True")
                raise ValueError("OpenAI API key is required when describe_images is True.")
            logger.debug("Initializing OpenAI client")
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.vision_model = vision_model

    def _create_parser(self) -> LlamaParse:
        """Helper method to create and configure the LlamaParse instance."""
        logger.debug("Creating LlamaParse instance")
        return LlamaParse(
            api_key=self.api_key,
            parse_mode=self.parse_mode,
            model=self.model,
            high_res_ocr=self.high_res_ocr,
            adaptive_long_table=self.adaptive_long_table,
            outlined_table_extraction=self.outlined_table_extraction,
            output_tables_as_HTML=self.output_tables_as_html,
            verbose=self.verbose,
            num_workers=self.num_workers,
        )

    def _extract_images(self, result) -> None:
        """Helper method to extract and save images from the parsed result."""
        logger.info("Extracting images from parsed result")
        try:
            result.get_image_documents(
                include_screenshot_images=self.include_screenshot_images,
                include_object_images=self.include_object_images,
                image_download_dir=self.image_dir,
            )
            logger.debug(f"Images extracted and saved to {self.image_dir}")
        except Exception as e:
            logger.error(f"Failed to extract images: {e}")

    def _encode_image(self, image_path: str) -> str:
        """Helper method to base64 encode an image."""
        logger.debug(f"Encoding image: {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                logger.debug(f"Successfully encoded image: {image_path}")
                return encoded
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise

    def _generate_image_description(self, image_path: str) -> str:
        """Helper method to generate a description/summary for an image using OpenAI's vision model."""
        if not self.describe_images:
            logger.debug("Image description generation skipped as describe_images is False")
            return ""
        logger.info(f"Generating description for image: {image_path}")
        try:
            base64_image = self._encode_image(image_path)
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
            )
            description = response.choices[0].message.content.strip()
            logger.debug(f"Generated description for {image_path}: {description[:50]}...")
            return description
        except Exception as e:
            logger.error(f"Error generating description for {image_path}: {e}")
            return "Image description not available."

    def _replace_placeholders_with_descriptions(self, page_content: str, images: list[str]) -> tuple[str, list[str]]:
        """Helper method to append image descriptions below placeholders in markdown."""
        logger.debug(f"Processing {len(images)} images for description replacement")
        descriptions = []
        if images:
            logger.info("Appending image descriptions to page content")
            page_content += "\n **This page contains images. Below are the detailed descriptions of each image, referenced by index:**"
            for index, img_path in enumerate(images):
                desc = self._generate_image_description(img_path)
                logger.info(f"Description for image {index} ({img_path}): {desc[:50]}...")
                page_content += f"\nImage:{index}, Description: {desc}"
                descriptions.append(desc)
        else:
            logger.debug("No images found for description replacement")
        return page_content, descriptions

    def _build_document(self, page, index: int) -> Document:
        """Helper method to build a LangChain Document from a single parsed page."""
        logger.debug(f"Building document for page {index + 1}")
        metadata = {
            'page': index + 1,
            'image_dir': self.image_dir,
            'text': page.text,
        }
        page_content = page.md  # Use markdown as the primary content

        # Collect image paths if images are present, but only if the file exists
        images = (
            [path for path in [os.path.join(self.image_dir, img.name) for img in page.images] if os.path.exists(path)]
        )
        logger.info(f"Found {len(images)} images on page {index + 1}: {images}")
        if images:
            metadata['images'] = images
            metadata['has_chart_or_image'] = True
            if self.describe_images:
                page_content, descriptions = self._replace_placeholders_with_descriptions(page_content, images)
                metadata['image_descriptions'] = descriptions
                logger.debug(f"Appended {len(descriptions)} image descriptions to page {index + 1}")
        else:
            metadata['has_chart_or_image'] = False
            logger.debug(f"No images found on page {index + 1}")

        logger.debug(f"Document built for page {index + 1} with metadata: {metadata}")
        return Document(page_content=page_content, metadata=metadata)

    def load(self) -> list[Document]:
        """
        Load and parse the PDF file, returning a list of LangChain Documents.

        :return: List of Document objects, one per page.
        """
        logger.info(f"Starting to load and parse PDF: {self.pdf_path}")
        # Create the parser instance
        parser = self._create_parser()
        # Parse the PDF file
        logger.debug("Parsing PDF file")
        try:
            result = parser.parse(self.pdf_path)
            logger.info("PDF parsing completed successfully")
        except Exception as e:
            logger.error(f"Failed to parse PDF {self.pdf_path}: {e}")
            raise
        # Extract and save images
        self._extract_images(result)
        # Initialize list to hold documents
        documents = []
        # Iterate over each parsed page and build documents
        logger.debug(f"Processing {len(result.pages)} pages")
        for index, page in enumerate(result.pages):
            doc = self._build_document(page, index)
            documents.append(doc)
            logger.info(f"Document created for page {index + 1}")
        logger.info(f"Loaded {len(documents)} documents from PDF")
        return documents

# Example usage with logging
if __name__ == "__main__":
    logger.info("Starting example usage of LlamaParseLoader")
    try:
        loader = LlamaParseLoader(
            "C:/Users/own/Downloads/Learn-RAG-code-only/Learn-RAG-code-only/pdf_files/transformer.pdf",
            api_key="YourLlamaparseapi_key",
            openai_api_key="Youropenai_api_key",
            describe_images=True,
            image_dir="C:/Users/own/Downloads/Learn-RAG-code-only 2_Updated_Code/Learn-RAG-code-only/extracted_images")
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents")
        # Print documents (for debugging; can be removed or replaced with logger)
        for i, doc in enumerate(documents):
            logger.debug(f"Document {i + 1}: {doc.page_content[:100]}... (Metadata: {doc.metadata})")
    except Exception as e:
        logger.error(f"Error in example usage: {e}")