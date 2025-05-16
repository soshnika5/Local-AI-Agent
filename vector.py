from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os, pandas as pd

"""
Use OllamaEmbedding for embedding and vectorizing documents.
Database will be hosted locally using Chomadb to allow quick retrieval for relevant information, 
pass to our model, and give more contextuallly relevant answers.
"""

### Load the data
df = pd.read_csv("realistic_restaurant_reviews.csv")

### Define embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

### Store vector database
db_location = "./chrome_langchain.db"

### Check if the database already exists
add_documents = not os.path.exists(db_location)

### If the database does not exist, prepare all the data by converting it to documents
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content= row["Title"] + "\n" + row["Review"],
            metadata={"rating":row["Rating"], 
                      "date": row["Date"]},
                      id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

### Initialize the vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

### if database does not exist, add documents to the vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

### Create retriever from the vector store to search for relevant documents (e.g. 5 most relevant reviews)
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5}
)
