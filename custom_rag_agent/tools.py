from langchain_classic.tools.retriever import create_retriever_tool
from data_ingestion import retriever

# Retrieve tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts, this returns only information about LLMs and AI",
)