import numpy as np
import matplotlib.pyplot as plt

# 시그모이드 함수
# 시그모이드 함수는 기울기 소실 문제가 발생하여 주로 이진 분류를 위해 출력층에서 사용됨
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ---------------------------------- 기본
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

# ---------------------------------- 그래프의 경사도 변형
# x = np.arange(-5.0, 5.0, 0.1)
# y1 = sigmoid(0.5*x)
# y2 = sigmoid(x)
# y3 = sigmoid(2*x)

# plt.plot(x, y1, 'r', linestyle='--') # w의 값이 0.5일때
# plt.plot(x, y2, 'g') # w의 값이 1일때
# plt.plot(x, y3, 'b', linestyle='--') # w의 값이 2일때
# plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
# plt.title('Sigmoid Function')
# plt.show()

# ---------------------------------- bias 변형
# x = np.arange(-5.0, 5.0, 0.1)
# y1 = sigmoid(x+0.5)
# y2 = sigmoid(x+1)
# y3 = sigmoid(x+1.5)

# plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
# plt.plot(x, y2, 'g') # x + 1
# plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
# plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
# plt.title('Sigmoid Function')
# plt.show()