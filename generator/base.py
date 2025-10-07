from typing import List

from langchain_core.documents import Document


class GeneratorBase:
    """
    Abstract base class for generating output using an LLM based on retrieved documents and a question.
    Subclasses must implement the _generate method to handle LLM-specific generation.
    """

    def _frame_prompt(self, docs: List[Document], question: str) -> str:
        """
        Frame a prompt by combining context from retrieved documents and the question.

        :param docs: List of retrieved Document objects.
        :param question: The user's question.
        :return: The framed prompt string.
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
Based on the following context, answer the question:

Context:
{context}

Question: {question}

Answer:
"""
        return prompt