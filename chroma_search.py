import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
client = chromadb.PersistentClient()
collection = client.get_or_create_collection('test')
df = pd.read_csv("./ChatbotData.csv")
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
if collection.count() == 0:
    print(">>>>>> DB is Empty <<<<<<<")
    docs = df["Q"].tolist()
    ids = [str(x) for x in df.index.tolist()]
    ## make embedding
    embeddings = default_ef(docs)
    print("######Start add collection#####")
    collection.add(
        embeddings=embeddings,
        documents=docs,
        ids = ids
    )

query = input("문장을 입력하세요 >>>> ")
queryText = default_ef([query])
results = collection.query(
    query_embeddings=queryText,
    n_results=10
)
indices = results['ids']
temp = df.iloc[indices[0]]
print (temp)