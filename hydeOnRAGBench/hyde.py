from llm import generate
from config import HYDE_PROMPT, ANSWER_PROMPT


def generate_hypothetical_doc(query):
    """Step 1: generate fake document for the query."""
    return generate(HYDE_PROMPT.format(query=query))


def generate_answer(query, retrieved_docs):
    """Step 4: generate answer from retrieved documents."""
    context = "\n\n".join(retrieved_docs)
    return generate(ANSWER_PROMPT.format(context=context, query=query))
