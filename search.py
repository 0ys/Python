import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
INDEX_FILE = "./sts.index"
#변환기 불러오기
#embedder = SentenceTransformer("Huffon/sentence-klue-roberta-base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
#데이터 불러오기
df = pd.read_csv("./ChatbotData.csv")

# faiss 계산하기
if os.path.exists(INDEX_FILE):
    #index파일이 존재하면 이것만 읽어들인다.
    index = faiss.read_index(INDEX_FILE)
else:
    #임베딩 벡터 불러오기
    embeddings = np.load("./embeddings.npy")
    print(embeddings.shape) # 임베딩 쉐이프 확인
    index = faiss.IndexFlatL2(embeddings.shape[1]) # 초기화 : 벡터의 크기를 지정
    index.add(embeddings) # 임베딩을 추가
    print(index.ntotal)
    faiss.write_index(index,"./sts.index")
top_k = 100
query = input("문장을 입력하세요 >>> ")
query_embedding = embedder.encode(query,normalize_embeddings=True, convert_to_tensor=True)
distances, indices = index.search(np.expand_dims(query_embedding,axis=0),top_k)
# 결과 확인
temp = df.iloc[indices[0]]
print(temp[['Q','A','label']].head(10))