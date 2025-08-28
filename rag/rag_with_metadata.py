import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

# --- Setup ---
load_dotenv()
embeddings = OpenAIEmbeddings()

# --- Simplified Document List ---
# The 'genre' and 'awards' metadata fields have been completely removed.
docs = [
    Document(
        page_content="In a theme park on a remote island, a wealthy entrepreneur secretly creates a zoo of living dinosaurs cloned from prehistoric DNA. Before the park can open to the public, a security breakdown allows the deadly creatures to escape and hunt the human visitors.",
        metadata={
            "year": 1993, "rating": 7.7, "director": "Steven Spielberg",
            "runtime_minutes": 127, "country": "USA", "is_sequel": False,
        },
    ),
    Document(
        page_content="A skilled thief, who steals information by entering people's dreams, is offered a chance to have his criminal history erased. To do so, he must perform the difficult task of 'inception': planting an idea into a target's subconscious.",
        metadata={
            "year": 2010, "director": "Christopher Nolan", "rating": 8.8,
            "runtime_minutes": 148, "country": "USA", "is_sequel": False,
        },
    ),
    Document(
        page_content="A revolutionary new psychotherapy treatment allows a detective to enter the dreams of her patients. When a prototype device is stolen, she must enter a shared dream world to find it, where reality and dreams start to merge. The film is a notable work of anime.",
        metadata={
            "year": 2006, "director": "Satoshi Kon", "rating": 8.6,
            "runtime_minutes": 90, "country": "Japan", "is_sequel": False,
        },
    ),
    Document(
        page_content="This adaptation of the classic novel follows the lives of the four March sisters—Meg, Jo, Beth, and Amy—as they come of age in America in the aftermath of the Civil War. It explores their struggles, ambitions, and enduring bond.",
        metadata={
            "year": 2019, "director": "Greta Gerwig", "rating": 7.8,
            "runtime_minutes": 135, "country": "USA", "is_sequel": False,
        },
    ),
    Document(
        page_content="A cowboy doll named Woody becomes profoundly threatened and jealous when a new spaceman figure, Buzz Lightyear, supplants him as the top toy in a boy's room. This animated family comedy follows their adventure.",
        metadata={
            "year": 1995, "runtime_minutes": 81, "director": "John Lasseter",
            "rating": 8.3, "country": "USA", "is_sequel": False,
        },
    ),
    Document(
        page_content="A guide, known as the 'Stalker', leads two clients—a writer and a professor—into a mysterious and forbidden territory known as the Zone. Within the Zone, there is a room that is said to grant the innermost wishes of anyone who enters it. This is a science fiction art house film.",
        metadata={
            "year": 1979, "director": "Andrei Tarkovsky", "rating": 8.1,
            "runtime_minutes": 162, "country": "Soviet Union", "is_sequel": False,
        },
    ),
    Document(
        page_content="A computer hacker named Neo discovers that his reality is a simulation created by sentient machines. He joins a rebellion to fight against the machines, learning to manipulate the simulated world in ways he never thought possible. An action and science fiction story.",
        metadata={
            "year": 1999, "director": "The Wachowskis", "rating": 8.7,
            "runtime_minutes": 136, "country": "USA", "is_sequel": False,
        },
    ),
    Document(
        page_content="The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption. This crime drama is known for its eclectic dialogue, ironic mix of humor and violence, and nonlinear narrative.",
        metadata={
            "year": 1994, "director": "Quentin Tarantino", "rating": 8.9,
            "runtime_minutes": 154, "country": "USA", "is_sequel": False,
        },
    ),
    Document(
        page_content="During her family's move to the suburbs, a 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts. She must work in a bathhouse for the spirits to find a way to free herself and her parents. An animated fantasy film.",
        metadata={
            "year": 2001, "director": "Hayao Miyazaki", "rating": 8.6,
            "runtime_minutes": 125, "country": "Japan", "is_sequel": False,
        },
    ),
    Document(
        page_content="The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal, and other historical events unfold through the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart. A romantic drama.",
        metadata={
            "year": 1994, "director": "Robert Zemeckis", "rating": 8.8,
            "runtime_minutes": 142, "country": "USA", "is_sequel": False,
        },
    ),
     Document(
        page_content="A meek hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron. They must face perilous trials and internal conflict as the forces of darkness hunt them. A fantasy adventure.",
        metadata={
            "year": 2001, "director": "Peter Jackson", "rating": 8.8,
            "runtime_minutes": 178, "country": "New Zealand", "is_sequel": False,
        },
    ),
    Document(
        page_content="Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan. The Kims cleverly install themselves as servants in the Park household, but their deception is threatened by an unexpected discovery. A thriller comedy.",
        metadata={
            "year": 2019, "director": "Bong Joon Ho", "rating": 8.5,
            "runtime_minutes": 132, "country": "South Korea", "is_sequel": False,
        },
    ),
]
vectorstore = Chroma.from_documents(docs, embeddings)

# --- Simplified Metadata Schema Definition ---
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI

metadata_field_info = [
    AttributeInfo( name="year", description="The year the movie was released", type="integer"),
    AttributeInfo( name="director", description="The name of the movie director", type="string"),
    AttributeInfo( name="rating", description="A 1-10 rating for the movie", type="float"),
    AttributeInfo( name="runtime_minutes", description="The total runtime of the movie in minutes", type="integer"),
    AttributeInfo( name="country", description="The country where the movie was produced", type="string"),
    AttributeInfo( name="is_sequel", description="A boolean flag that is True if the movie is a sequel, and False otherwise", type="boolean"),
]

document_content_description = "A detailed synopsis of a movie, including its plot, main characters, and themes."
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# --- Adjusted Query Examples ---

# print("\n\n--- QUERY 1: Combining semantic search with a country filter ---")
# # This query now relies purely on semantic search for "class struggle"
# # while filtering on the country metadata.
# response = retriever.invoke("Tell me about a non-American movie dealing with class discrimination.")
# print(f"Found {len(response)} document(s).\n")
# for doc in response:
#     print(f"Movie: {doc.metadata['director']}'s film from {doc.metadata['year']} ({doc.metadata['country']})")
#     print(f"Content: {doc.page_content[:100]}...")
#     print("-" * 20)

# print("\n\n--- QUERY 2: Combining numerical and director filters (Unaffected) ---")
# # This query is unaffected as it doesn't use the removed fields.
# response = retriever.invoke("Find me a movie by Andrei Tarkovsky that is longer than 160 minutes.")
# print(f"Found {len(response)} document(s).\n")
# for doc in response:
#     print(f"Movie: {doc.metadata['director']}'s film from {doc.metadata['year']}")
#     print(f"Runtime: {doc.metadata['runtime_minutes']} minutes")
#     print("-" * 20)
#
#
# print("\n\n--- QUERY 3: Using boolean logic with semantic search for genre ---")
# # This query now relies on the words 'animated' and 'spirits' being in the
# # document content, while still filtering for `is_sequel: False`.
# response = retriever.invoke("I'm looking for a non-sequel animated film about spirits or magic.")
# print(f"Found {len(response)} document(s).\n")
# for doc in response:
#     print(f"Movie from {doc.metadata['year']} ({doc.metadata['country']})")
#     print(f"Director: {doc.metadata['director']}")
#     print(f"Content: {doc.page_content[:100]}...")
#     print("-" * 20)