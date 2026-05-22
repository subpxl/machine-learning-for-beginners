import matplotlib.pyplot as plt
import numpy as np
x = [1,2,3,4]
y = [2,4,6,8]

# fig, ax = plt.subplots()

# ax.plot(    x,
#     y,
#     color="red",
#     linewidth=3,
#     linestyle="--",
#     marker="o")
# ax.set_xlabel("X Values")
# ax.set_ylabel("Y Values")
# ax.set_title("Simple Plot")
# ax.plot(x, y, label="Data")
# ax.scatter(x, y)
# ax.legend()
# ax.grid(True)
# plt.show()

# data = np.random.normal(50, 10, 1000)

# plt.hist(data, bins=30)

# plt.show()


# subjects = ["Math", "Physics", "Biology"]
# marks = [85, 90, 48]
# plt.bar(subjects, marks)

# plt.show()


x = [1,2,3,4,]
y1=[1,4,9,16]
y2=[1,2,3,4]
y3=[1,5,6,8]

fig, ax = plt.subplots(2, 2)

plt.plot(x, y1, label="Square")
plt.plot(x, y2, label="Linear")
plt.plot(x, y3, label="new line")
plt.legend()
plt.grid(True)
plt.show()


x = np.linspace(-10,10,500)
y= x**2

plt.plot(x,y)
plt.show()

y = np.sin(x)
plt.plot(x, y)

plt.savefig("graph.png")
plt.show()


from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=300,
    centers=3,
    random_state=42
)

plt.scatter(X[:,0], X[:,1], c=y)

plt.show()