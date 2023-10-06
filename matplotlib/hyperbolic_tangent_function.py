import numpy as np
import matplotlib.pyplot as plt

# 하이퍼볼릭탄젠트 함수(tanh)는 입력값을 -1과 1사이의 값으로 변환함
x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y = np.tanh(x)

plt.plot(x, y)
plt.plot([0,0],[1.0,-1.0], ':')
plt.axhline(y=0, color='orange', linestyle='--')
plt.title('Tanh Function')
plt.show()