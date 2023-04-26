from sklearn.datasets import load_breast_cancer
import problexity as px
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7,7))
X, y = load_breast_cancer(return_X_y=True)

# Initialize CoplexityCalculator with default parametrization
cc = px.ComplexityCalculator()

# Fit model with data
cc.fit(X,y)
print(cc.complexity)
print(cc._metrics())
print(cc.score())
print(cc.report())
cc.plot(fig, (1,1,1))

plt.tight_layout()
plt.savefig("problexity.png")