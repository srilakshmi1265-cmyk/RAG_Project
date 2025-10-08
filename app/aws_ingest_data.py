#This is the file used for ingest image in docker
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import boto3
import os
import tempfile
from extract.LLamaParse.extract_page import LlamaParseLoader
from store.aws_opensearch import OpenSearchStore
from embed.titan.embed import TitanTextImageEmbeddings
import logging
from store.base import *
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from embed.clip.embed import CLIPEmbeddings
from embed.titan.embed import TitanTextEmbeddings, TitanTextImageEmbeddings
from extract.LLamaParse.extract_page import LlamaParseLoader

# --- Load AWS credentials from environment ---
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
#llama_parse_key = os.getenv("LLAMA_PARSE_KEY")
opensearch_domain_url = os.getenv("OPENSEARCH_DOMAIN_URL")
index_name =os.getenv("INDEX_NAME")
# --- S3 client ---
client = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# --- FastAPI app ---
app = FastAPI(title="Importing The Documents")

class QueryRequest(BaseModel):
    #file_key: str  # S3 object key (e.g. "pdfs/transformer.pdf")
    pass
@app.post("/ingest")
async def opensearchdoc():
    bucket = "sriluragpdffiles"
    key = "pdf/9.pdf"

    # --- Download file to a temporary location ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        client.download_fileobj(bucket, key, tmp_file)
        tmp_path = tmp_file.name

    # --- Create temp directory for extracted images ---
    image_dir = tempfile.mkdtemp(prefix="extracted_images_")

    # --- Load and parse PDF ---
    loader = LlamaParseLoader(
        tmp_path,"llx-VOvLvSpXbxnlKgg7U4wiHCROSYbEsH8JCQvgrQw6fD5JZ5fM",
        openai_api_key=openai_api_key,
        describe_images=True,
        image_dir=image_dir
    )
    docs = await loader.aload()

    # --- Titan embeddings ---
    titan_embeddings = TitanTextImageEmbeddings(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # --- Store in OpenSearch ---
    vectorstore = OpenSearchStore(loader, titan_embeddings)
    vectorstore.store(opensearch_url=opensearch_domain_url,
                      index_name=index_name,
    http_auth=("Self_Admin123", "Srilu@123"),  # master user creds
        use_ssl=True,
        verify_certs=True,
    )

    return {
        "message": f"File {key} ingested successfully",
        #"index": index,
        #"image_dir": image_dir
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)