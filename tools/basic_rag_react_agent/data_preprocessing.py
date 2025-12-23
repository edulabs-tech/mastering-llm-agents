from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings



# Load environment variables from .env file
load_dotenv()


print(f"--------- Creating Vectorstore with mortgage data --------")

# INDEXING: LOAD
loader = PyPDFLoader("../../docs/59321_booklet_guide_mashknta_A4_Pages_03.pdf",)
docs = loader.load()

# INDEXING: SPLIT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# INDEXING: STORE
VECTORSTORE = Chroma.from_documents(
    documents=all_splits,
    embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
)

retriever = VECTORSTORE.as_retriever(search_kwargs={"k": 3})

print(f"--------- DONE --------")