import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
df = pd.read_csv("./ChatbotData.csv") #from https://github.com/songys/Chatbot_data
sentences = df['Q'].to_list()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(sentences)
np.save("./embeddings.npy", embeddings)