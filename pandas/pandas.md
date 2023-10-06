# Pandas

pandas는 총 세 가지 데이터 구조를 사용함
- 시리즈(Series)
- 데이터프레임(DataFrame)
- 패널(Panel)

### Series
시리즈 클래스는 1차원 배열의 값(values)에 각 값에 대응되는 인덱스(index)를 부여할 수 있는 구조를 가짐
```python
sr = pd.Series([17000, 18000, 1000, 5000], index=["피자", "치킨", "콜라", "맥주"])
```
시리즈의 값 : [17000 18000  1000  5000]
시리즈의 인덱스 : Index(['피자', '치킨', '콜라', '맥주'], dtype='object')

### DataFrame
데이터프레임은 2차원 리스트를 매개변수로 전달함, 즉 행과 열을 가지는 자료구조임
시리즈가 인덱스와 값으로 구성된다면, 데이터프레임은 열(columns), 인덱스, 값으로 구성됨
```python
data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]
df = pd.DataFrame(data, columns=['학번', '이름', '점수'])
```
#### 외부데이터 읽기
```python
df = pd.read_csv('example.csv')
```
#### 간략한 정보 파악
```python
df.info()
```
#### 데이터 조회
```python
df.head(3)
df.tail(3)
df['학번']
```
#### 불필요한 데이터 삭제
```python
df.drop(['column1', 'column2'], axis=1) # 원본은 수정하지 않음
train_df.drop(['Pclass'], axis=1, inplace=True) # 원본 객체를 수정
```
#### unique 데이터 개수 세기
```python
df['column1'].value_counts()
```
#### 원-핫 인코딩
```python
pclass_train_dummies = pd.get_dummies(train_df['Pclass'])
```
#### column명 지정
```python
dummies.columns = ['column1', 'column2', 'column3']
```
#### df 합성
```python
train_df = train_df.join(train_dummies)
```
#### 비어있는 데이터 채우기
```python
df['column1'].fillna(df['column1'].mean(), inplace=True)
df['column1'].fillna(0, inplace=True)
```
