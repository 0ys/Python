import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# NLTK 다운로드
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# 데이터 로드
data = pd.read_csv('review_data.csv')

# 텍스트 전처리 함수
def preprocess_text(text):
    text = text.lower()  # 소문자로 변환
    text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
    text = re.sub(r'[\W_]+', ' ', text)  # 구두점 및 특수 문자 제거
    text = text.strip()  # 불필요한 공백 제거
    return text

# 불용어 제거 및 어간 추출
def tokenize_and_stem(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# 데이터 전처리 및 변환
data['processed_text'] = data['review'].apply(preprocess_text)
data['processed_text'] = data['processed_text'].apply(tokenize_and_stem)

# TF-IDF 벡터 생성
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['processed_text'])

# 단어 빈도 분석
def get_top_n_words(corpus, n=10):
    words = ' '.join(corpus).split()
    word_freq = Counter(words)
    return word_freq.most_common(n)

sentiment_categories = data['sentiment'].unique()
word_frequencies = {}

for sentiment in sentiment_categories:
    sentiment_corpus = data[data['sentiment'] == sentiment]['processed_text']
    word_frequencies[sentiment] = get_top_n_words(sentiment_corpus, 10)

# 단어 빈도 시각화
for sentiment, freq in word_frequencies.items():
    words, counts = zip(*freq)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(words), y=list(counts))
    plt.title(f"Top Words in {sentiment} Sentiment Reviews")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

from wordcloud import WordCloud

# 워드 클라우드 생성
for sentiment, freq in word_frequencies.items():
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(dict(freq))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {sentiment} Sentiment Reviews")
    plt.show()

# 리뷰 길이 분석
data['review_length'] = data['review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment', y='review_length', data=data)
plt.title("Review Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Review Length")
plt.show()

# 분석 결과 요약
def summarize_results():
    print("\n### Sentiment Analysis Summary ###")
    for sentiment, freq in word_frequencies.items():
        print(f"\n{sentiment.capitalize()} Sentiment:")
        for word, count in freq:
            print(f"  {word}: {count}")

summarize_results()
