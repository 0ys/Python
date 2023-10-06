import numpy as np
import matplotlib.pyplot as plt

# 소프트맥스 함수(Softmax function)
# 은닉층에서는 ReLU(또는 ReLU 변형) 함수들을 사용하는 것이 일반적인 반면, 
# 출력층에서는 소프트맥스 함수나 시그모이드 함수를 사용함
# 시그모이드 함수: 이진 분류 (Binary Classification) 문제에 사용
# 소프트맥스 함수: 다중 클래스 분류(MultiClass Classification) 문제에 사용

x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y = np.exp(x) / np.sum(np.exp(x))

plt.plot(x, y)
plt.title('Softmax Function')
plt.show()