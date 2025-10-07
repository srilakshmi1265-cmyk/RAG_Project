# Purpose:ge_aws_bedrock ->
# Flow explained:
# Take a query ("What is Self Attention").
# Convert query + docs into embeddings via Titan.
# Fetch most relevant documents from OpenSearch.
# Feed them into Claude (Bedrock LLM).
# Get a summarized, contextual answer.
# The BedrockGenerator class is a wrapper around AWS Bedrock’s LLMs (Anthropic Claude, Meta Llama, etc.).
# Purpose:ge_aws_bedrock ->
# It:Connects to AWS Bedrock securely using boto3.
# Accepts a question + a list of context documents.
# Frames the context into a prompt.
# Calls the Bedrock model (BedrockChat) to generate an answer.
# Returns the summarized/generated output.
# So basically:
# Retriever (OpenSearch) fetches relevant docs →
# BedrockGenerator adds the docs into a prompt →
# Claude model on Bedrock generates a natural language answer.

import os
import sys

import boto3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))
from typing import List


from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document

# from embed.AmazonTitan.amazon_embed import TitanTextImageEmbeddings, TitanTextEmbeddings
from embed.titan.embed import TitanTextEmbeddings, TitanTextImageEmbeddings
from generator.base import GeneratorBase
from retriever.re_opensearch import OpenSearchRetriever

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


class BedrockGenerator(GeneratorBase):
    """
    A generator class using AWS Bedrock for output generation, extending GeneratorBase.
    Supports models like Anthropic Claude or Meta Llama via BedrockChat.
    """

    def __init__(
        self,
        aws_access_key_id: str = AWS_ACCESS_KEY_ID,
        aws_secret_access_key: str = AWS_SECRET_ACCESS_KEY,
        region_name: str = "us-east-1",
        # model_id: str = "anthropic.claude-v2",  # Example: Claude v2; adjust as needed
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
    ):
        """
        Initialize the BedrockGenerator.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param region_name: AWS region (default: "us-east-1").
        :param model_id: Bedrock model ID (default: "anthropic.claude-v2").
        """
        # self.llm = BedrockChat(
        #     model_id=model_id,
        #     region_name=region_name,
        #     credentials_profile_name=None,  # Use explicit keys
        #     aws_access_key_id=aws_access_key_id,
        #     aws_secret_access_key=aws_secret_access_key,
        # )
        boto3_session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=region_name,
        )

        self.llm = BedrockChat(
            model_id=model_id, client=boto3_session.client("bedrock-runtime")
        )

    def generate_output(self, docs: List[Document], question: str) -> str:
        """
        Generate output using Bedrock based on retrieved documents and the question.

        :param docs: List of retrieved Document objects.
        :param question: The user's question.
        :return: The generated answer string.
        """
        prompt = self._frame_prompt(docs, question)
        response = self.llm.invoke(prompt)
        return response.content.strip()


if __name__ == "__main__":
    query = "What is Self Attention"
    embeddings = TitanTextImageEmbeddings()  # model_id="amazon.titan-embed-text-v1")
    print("Embeddings are working for query")
    # run_manager = CallbackManagerForRetrieverRun()
    vectorstore_for_OSVS = OpenSearchVectorSearch(
        opensearch_url="https://search-clinicaldatadomain-xqpcko5eaq5dnxxc4okbwpj2uy.aos.us-east-1.on.aws/",
        index_name="ragtitan",
        embedding_function=embeddings,
        http_auth=("Self_Admin123", "Srilu@123"),
    )
    os_retriever = OpenSearchRetriever(vectorstore=vectorstore_for_OSVS, top_k=4)
    print("Documents are Fetched Successfully")
    bedrock = BedrockGenerator(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    print("BedRock")
    retrieved_docs = os_retriever.get_relevant_documents(query=query)
    print("Getting relevant Documents")
    answer = bedrock.generate_output(retrieved_docs, query)
    print("Answer Generated Successfully")
    print("Summary generated" + answer)
