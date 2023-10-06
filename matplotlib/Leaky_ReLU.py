import numpy as np
import matplotlib.pyplot as plt

# Leaky ReLU
# f(x)=max(ax,x)
# 입력값이 음수일 경우에 0이 아니라 0.001과 같은 매우 작은 수를 반환
# a는 하이퍼파라미터로 Leaky('새는') 정도를 결정하며 일반적으로는 0.01의 값을 가짐
# (여기서 말하는 '새는 정도'라는 것은 입력값의 음수일 때의 기울기를 비유)

a = 0.1

def leaky_relu(x):
    return np.maximum(a*x, x)

x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Leaky ReLU Function')
plt.show()