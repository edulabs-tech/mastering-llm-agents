from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever


# Load environment variables from .env file
load_dotenv()


embedding_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# note: try "weaker" model - gemini-2.0-flash and see the error


docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "action"},
    ),
    Document(
        page_content="A fight club that is not a fight club, but is a fight club",
        metadata={"year": 1994, "rating": 8.7, "genre": "action"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "genre": "thriller", "rating": 8.2},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "rating": 8.3, "genre": "drama"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={"year": 1979, "rating": 9.9, "genre": "science fiction"},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "genre": "thriller", "rating": 9.0},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated", "rating": 9.3},
    ),
    Document(
        page_content="The toys come together to save their friend from a kid who doesn't know how to play with them",
        metadata={"year": 1997, "genre": "animated", "rating": 9.1},
    ),
]

vectorstore = Chroma.from_documents(docs, embedding_model)

# no metadata
# retriever = vectorstore.as_retriever(
#     search_type="similarity", 
#     search_kwargs={"k": 4}
# )

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="rating", 
        description="A 1-10 rating for the movie", 
        type="integer"
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm, 
    vectorstore, 
    document_contents="Brief summary of a movie", 
    metadata_field_info=metadata_field_info, 
    verbose=True
)


response = retriever.invoke("What are some highly rated movies (above 9)?")
print(response)

print("\n------------\n")

response = retriever.invoke("I want to watch a movie about toys rated higher than 9")
print(response)

print("\n------------\n")

response = retriever.invoke("What's a highly rated (above or equal 9) thriller film?")
print(response)

print("\n------------\n")

response = retriever.invoke("What's a movie after 1990 but before 2005 that's all about dinosaurs, and preferably has the action genre")
print(response)