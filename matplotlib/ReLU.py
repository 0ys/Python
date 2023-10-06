import numpy as np
import matplotlib.pyplot as plt

# 렐루 함수
# f(x)=max(0,x)
# 음수를 입력하면 0을 출력하고, 양수를 입력하면 입력값을 그대로 반환
# 출력값이 특정 양수값에 수렴하지 않으며, 0 이상의 입력값의 경우에는 미분값이 항상 1임
# 깊은 신경망의 은닉층에서 잘 작동하며, 연산 속도도 빠름

# 죽은 렐루(dying ReLU) 문제: 
# 입력값이 음수면 기울기. 즉, 미분값도 0이 되어 이 뉴런은 다시 회생하기 어려움
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Relu Function')
plt.show()