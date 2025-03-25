import numpy as np
# 将输出重定向到txt文件
import sys

# 打开一个txt文件，并将sys.stdout指向该文件
for ll in range(10001, 6, -1):
    with open(f'测试/{ll}.txt', 'w') as f:
        sys.stdout = f  # 重定向print的输出
        l = ll
        n = 50
        m = 30
        print(l)
        size = 0
        for i in range(l):
            size_j = np.random.randint(21)
            size += size_j
            print(i + 1, np.random.randint(10, 601), size_j, end=" ")
            n_machine = np.random.randint(1, n)
            print(n_machine, end=" ")
            print(" ".join(np.random.choice(range(1, n+1), n_machine, replace=False).astype(str)))
        print(n)
        for i in range(n):
            print(i+1, np.random.randint(1, 21))
        print(m)
        for i in range(m):
            print(i, np.random.randint(1, 21), np.random.randint(int(size/m), 1050001))
