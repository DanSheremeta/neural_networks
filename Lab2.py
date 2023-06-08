import numpy as np
import matplotlib.pyplot as plt


def activation(a, f):
    return np.clip(a, 0, f)


def act(arr, k, i):
    summa = np.sum(arr) - arr[i]

    return arr[i] - k * summa


# функція пошуку найбільш схожого вектора-шаблону до вхідного вектора
def similar(e, y, f):
    a = 1.0 / 2

    e_pr = np.copy(e)
    # поки значення вектора e не стабілізувалося (збігається до попереднього значення), виконувати цикл
    while not np.allclose(e_pr, e):
        # копіювання значення вектора e для наступної ітерації
        e_pr = np.copy(e)
        # обчислення нового вектора y з використанням функції act та попереднього значення вектора e
        y = np.array([act(e_pr, a, i) for i in range(n)])
        # обчислення нового значення вектора e з використанням функції активації та нового вектора y
        e = activation(y + f, f)

    return np.argmax(e) + 1


n = 2
m = 16
rus = np.array([[1, -1, -1, -1,
                 1, -1, -1, -1,
                 1, -1, -1, -1,
                 1, 1, 1, 1],
                [1, 1, 1, 1,
                 1, -1, -1, 1,
                 1, 1, 1, 1,
                 1, -1, -1, -1]])
weight = rus / 2.0
f = m / 2.0
vector = np.array([1, 1, 1, 1,
                   1, -1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, -1, -1])
y = np.zeros(n)
e = np.zeros(n)

for i in range(n):
    y[i] = np.dot(weight[i], vector) + f
    e[i] = y[i]
plt.axis('off')
plt.imshow(vector.reshape((4, 4)), cmap='gray_r')
plt.show()
print("The vector is similar to vector number", similar(e, y, f), "with probability", e[similar(e, y, f) - 1] / m)
