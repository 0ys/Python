import numpy as np

# 1차원 배열
vec = np.array([1, 2, 3, 4, 5])
# print(vec)

# 2차원 배열
mat = np.array([[10, 20, 30], [ 60, 70, 80]]) 
# print(mat)

# print('vec의 타입 :',type(vec))
# print('vec의 축의 개수 :',vec.ndim) # 축의 개수 출력
# print('vec의 크기(shape) :',vec.shape) # 크기 출력

# print('mat의 타입 :',type(mat))
# print('mat의 축의 개수 :',mat.ndim) # 축의 개수 출력
# print('mat의 크기(shape) :',mat.shape) # 크기 출력

# --------------------------------------------------------------------

# 모든 값이 0인 2x3 배열 생성.
zero_mat = np.zeros((2,3))
# print(zero_mat)

# 모든 값이 1인 2x3 배열 생성.
one_mat = np.ones((2,3))
# print(one_mat)

# 모든 값이 특정 상수인 배열 생성. 이 경우 7.
same_value_mat = np.full((2,3), 7)
# print(same_value_mat)

# 대각선 값이 1이고 나머지 값이 0인 2차원 배열을 생성.
eye_mat = np.eye(5)
# print(eye_mat)

# 임의의 값으로 채워진 배열 생성
random_mat = np.random.random((2,2)) # 임의의 값으로 채워진 배열 생성
# print(random_mat)

# --------------------------------------------------------------------
# np.arrange(n)

# 0부터 9까지
range_vec = np.arange(10)
# print(range_vec)

# 1부터 9까지 +2씩 적용되는 범위
n = 3
range_n_step_vec = np.arange(1, 10, n)
# print(range_n_step_vec)

# --------------------------------------------------------------------
# np.reshape()
reshape_mat = np.array(np.arange(30)).reshape(5,6)
# print(reshape_mat)

# --------------------------------------------------------------------
# numpy 슬라이싱
mat = np.array([[1, 2, 3], [4, 5, 6]])
# print(mat)

# 첫번째 행 출력
slicing_mat = mat[0, :]
# print(slicing_mat)

# 두번째 열 출력
slicing_mat = mat[:, 1]
# print(slicing_mat)

# --------------------------------------------------------------------
# Numpy 정수 인덱싱(integer indexing)
mat = np.array([[1, 2], [4, 5], [7, 8]])
# print(mat)

# 1행 0열의 원소
# => 0부터 카운트하므로 두번째 행 첫번째 열의 원소.
# print(mat[1, 0])

# mat[[2행, 1행],[0열, 1열]]
# 각 행과 열의 쌍을 매칭하면 2행 0열, 1행 1열의 두 개의 원소.
indexing_mat = mat[[2, 1],[0, 1]]
# print(indexing_mat)

# --------------------------------------------------------------------
# Numpy 연산
x = np.array([1,2,3])
y = np.array([4,5,6])

# result = np.add(x, y)와 동일.
result = x + y
# print(result)

# result = np.subtract(x, y)와 동일.
result = x - y
# print(result)

# result = np.multiply(result, x)와 동일.
result = x * y
# print(result)

# result = np.divide(result, x)와 동일.
result = y / x
# print(result)

# 벡터의 행렬곱
mat1 = np.array([[1,2],[3,4]])
mat2 = np.array([[5,6],[7,8]])
mat3 = np.dot(mat1, mat2)
print(mat3)