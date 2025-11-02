from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables from .env file
load_dotenv()

# INDEXING: LOAD
loader = PyPDFLoader("../docs/59321_booklet_guide_mashknta_A4_Pages_03.pdf",)
docs = loader.load()

# INDEXING: SPLIT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# INDEXING: STORE
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
)

# RETRIEVAL AND GENERATION: RETRIEVAL
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# RETRIEVAL AND GENERATION: GENERATE
# Let’s put it all together into a chain that takes a question,
# retrieves relevant documents, constructs a prompt,
# passes it into a model, and parses the output.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)


def format_docs(original_docs):
    return "\n\n".join(doc.page_content for doc in original_docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def stream_rag_chain(text):
    for chunk in rag_chain.stream(text):
        yield chunk
