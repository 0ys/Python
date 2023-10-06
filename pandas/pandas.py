import pandas as pd

# Series
sr = pd.Series([17000, 18000, 1000, 5000], index=["피자", "치킨", "콜라", "맥주"])
# print('시리즈 출력 :')
# print('-'*15)
# print(sr)
# print('시리즈의 값 : {}'.format(sr.values))
# print('시리즈의 인덱스 : {}'.format(sr.index))

# --------------------------------------------------------------------

# DataFrame
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)

# print('데이터프레임 출력 :')
# print('-'*18)
# print(df)
# print('데이터프레임의 인덱스 : {}'.format(df.index))
# print('데이터프레임의 열이름: {}'.format(df.columns))
# print('데이터프레임의 값 :')
# print(df.values)

# 리스트로 생성하기
data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]

# df = pd.DataFrame(data)
# print(df)
df = pd.DataFrame(data, columns=['학번', '이름', '점수'])
print(df)

# 딕셔너리로 생성하기
data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
    '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
    }

# df = pd.DataFrame(data)
# print(df)

print('-'*15)
# 앞 부분을 3개만 보기
print(df.head(3))

print('-'*15)
# 뒷 부분을 3개만 보기
print(df.tail(3))

print('-'*15)
# '학번'에 해당되는 열을 보기
print(df['학번'])

# 외부데이터 읽기
# df = pd.read_csv('example.csv')
# print(df)