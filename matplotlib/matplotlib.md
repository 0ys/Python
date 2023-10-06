# Matplotlib
맷플롯립(Matplotlib)은 데이터를 차트(chart)나 플롯(plot)으로 시각화하는 패키지로, 분석 전 데이터 이해를 위한 시각화나, 데이터 분석 결과를 시각화하기 위해서 사용됨

## plot() 
plot()은 라인 플롯을 그림
plot()에 x축과 y축의 값을 기재하고 그림을 표시하는 show()를 통해서 시각화
```python
plt.title('test')
plt.plot([1,2,3,4],[2,4,8,6])
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```