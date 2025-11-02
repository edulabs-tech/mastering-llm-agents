from langchain_community.document_loaders import WebBaseLoader


loader = WebBaseLoader(
    ["https://angular.dev/guide/signals", 
     "https://angular.dev/guide/signals/linked-signal", 
     "https://angular.dev/guide/signals/resource"]
)

docs = loader.load()

print(len(docs))
print(docs)