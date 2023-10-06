# Numpy
넘파이(Numpy)는 수치 데이터를 다루는 파이썬 패키지fh, 다차원 행렬 자료구조인 ndarray를 통해 벡터 및 행렬을 사용하는 선형 대수 계산에서 주로 사용됨
`np.array()`는 리스트, 튜플, 배열로 부터 ndarray를 생성함
```python
vec = np.array([1, 2, 3, 4, 5])
```

## ndarray
- 축의 개수: `a.ndim`
- 배열 크기: `a.shape`

### 특정 원소를 가지는 배열
```python
zero_mat = np.zeros((2,3))
one_mat = np.ones((2,3))
same_value_mat = np.full((2,2), 7)
eye_mat = np.eye(5)
random_mat = np.random.random((2,2))
```

## np.arange()
`np.arange(n)`은 0부터 n-1까지의 값을 가지는 배열을 생성
`np.arange(i, j, k)`는 i부터 j-1까지 k씩 증가하는 배열을 생성
```python
range_vec = np.arange(10) # [0 1 2 3 4 5 6 7 8 9]
range_n_step_vec = np.arange(1, 10, 2) # [1 3 5 7 9]
```

## np.reshape()
```python
ndarray.reshape(n,m)
```

## numpy 슬라이싱
```python
slicing_mat = mat[0, :] # 첫번째 행
slicing_mat = mat[:, 1] # 두번째 열
```

## numpy 정수 인덱싱(integer indexing)
특정 위치의 원수만 가져옴
```python
a = mat[1, 0]
indexing_mat = mat[[2, 1],[0, 1]]
```

## numpy 연산
연산자 +, -, *, / 또는 np.add(), np.subtract(), np.multiply(), np.divide()를 사용
```python
result = np.add(x, y)
result = np.subtract(x, y)
result = np.multiply(result, x)
result = np.divide(result, x)
mat3 = np.dot(mat1, mat2)
```