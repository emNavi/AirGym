import torch
import matplotlib.pyplot as plt

# 定义参数方程
# t 的范围可以根据实际情况调整
t = torch.linspace(0, 4 * torch.pi, 1000)

# 定义 x 和 y 的函数
# x = 3 * torch.sin(t) / (1 + torch.cos(t) ** 2)
# y = 3 * torch.sin(t) * torch.cos(t) / (1 + torch.cos(t) ** 2)

x = 3 * torch.sin(t)
y = 3 * torch.sin(t) * torch.cos(t)


# 绘制参数方程图
plt.figure(figsize=(8, 8))
plt.plot(x.numpy(), y.numpy(), label="Parametric Curve", color='b')
plt.title("Parametric Curve: x=f(t), y=g(t)")
plt.xlabel("x = f(t)")
plt.ylabel("y = g(t)")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
