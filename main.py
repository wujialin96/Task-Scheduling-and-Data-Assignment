import numpy as np
import sys
import time
# from line_profiler import profile


def masked_argmin(x, condition):
    valid_idx = np.where(condition)[0]
    return valid_idx[x[valid_idx].argmin()]


class JobSchedulingDataAssignmentProblem:
    def __init__(self):
        self.start_time = time.time()
        self.log = []
        self.num = 0
        self.size = []
        self.data = []
        self.affinitive_machine = []
        self.process_time = []
        self.store_time = []
        self.total_time = []

        self.power = []
        self.speed = []
        self.capacity = []

        self.pre_data = {}
        self.after_data = {}
        self.pre_task = {}
        self.after_task = {}

        self.read_data()
        self.limit = np.array([len(self.pre_data[i]) + len(self.pre_task[i]) for i in range(self.num)])
        self.pre_count = self.limit.copy()
        self.arrive_time = np.zeros(self.num, dtype=int)
        self.a = np.zeros(self.num, dtype=int)
        self.c = {i: 0 for i in range(self.num)}
        self.d = np.zeros(self.num, dtype=int)
        self.r = np.zeros(self.num, dtype=int)
        self.task_machine = np.zeros(self.num, dtype=int) - 1
        self.task_disk = np.zeros(self.num, dtype=int) - 1
        self.machine_end_time = np.zeros(len(self.power), dtype=int)
        self.use = self.capacity.copy()
        self.time = 1
        self.count = 1

    # @profile
    def reset(self):
        self.log = []
        self.pre_count = self.limit.copy()
        self.arrive_time = np.zeros(self.num, dtype=int)
        self.a = np.zeros(self.num, dtype=int)
        self.c = np.zeros(self.num, dtype=int)
        self.d = np.zeros(self.num, dtype=int)
        self.r = np.zeros(self.num, dtype=int)
        self.task_machine = self.a - 1
        self.task_disk = self.a - 1
        self.machine_end_time = np.zeros(len(self.power), dtype=int)
        self.use = self.capacity.copy()

    def read_data(self, end_index=0):
        lines = sys.stdin.readlines()
        lines = [list(map(int, line.split())) for line in lines]
        num, self.num = lines[0][0], lines[0][0]
        start_index = end_index + 1
        end_index = start_index + num
        for line in lines[start_index:end_index]:
            self.size.append(line[1])
            self.data.append(line[2])
            affinitive_machine = line[4:]
            affinitive_machine = [i - 1 for i in affinitive_machine]
            self.affinitive_machine.append(affinitive_machine)

        num = lines[end_index][0]
        start_index = end_index + 1
        end_index = start_index + num
        for line in lines[start_index:end_index]:
            self.power.append(line[1])

        num = lines[end_index][0]
        start_index = end_index + 1
        end_index = start_index + num
        for line in lines[start_index:end_index]:
            self.speed.append(line[1])
            self.capacity.append(line[2])

        # 读取数据依赖关系
        pre_data = {i: [] for i in range(self.num)}
        after_data = {i: [] for i in range(self.num)}
        num = lines[end_index][0]
        start_index = end_index + 1
        end_index = start_index + num
        for x, y in lines[start_index:end_index]:
            pre_data[y - 1].append(x - 1)
            after_data[x - 1].append(y - 1)
        self.pre_data = pre_data
        self.after_data = after_data

        # 读取任务依赖关系
        pre_task = {i: [] for i in range(self.num)}
        after_task = {i: [] for i in range(self.num)}
        num = lines[end_index][0]
        start_index = end_index + 1
        end_index = start_index + num
        for x, y in lines[start_index:end_index]:
            pre_task[y - 1].append(x - 1)
            after_task[x - 1].append(y - 1)
        self.pre_task = pre_task
        self.after_task = after_task

        # 转换为数组
        self.power = np.array(self.power)
        self.speed = np.array(self.speed)
        self.capacity = np.array(self.capacity)
        self.data = tuple(self.data)
        self.affinitive_machine = tuple(self.affinitive_machine)

        # 预计算
        self.process_time = tuple([-(-size // self.power[affinitive_machine]) for size, affinitive_machine in zip(self.size, self.affinitive_machine)])
        self.store_time = np.array([-(-data // self.speed) for data in self.data])
        self.r = self.store_time.min(axis=1)
        self.total_time = np.array([p.sum() + s.sum() + self.r[a].sum() for p, s, a in zip(self.process_time, self.store_time, self.pre_data.values())])
        self.total_time = -self.total_time / self.total_time.max()

    def update_task(self, task_id):
        affinitive_machine = self.affinitive_machine[task_id]
        a = np.maximum(self.machine_end_time[affinitive_machine], self.arrive_time[task_id])
        c = a + self.r[task_id] + self.process_time[task_id]
        machine_id = c.argmin()
        d = c[machine_id] + self.store_time[task_id]
        disk_id = masked_argmin(d, self.use >= self.data[task_id])
        self.a[task_id], self.c[task_id], self.d[task_id] = a[machine_id], c[machine_id], d[disk_id]
        self.task_machine[task_id], self.task_disk[task_id] = affinitive_machine[machine_id], disk_id

    def solve_greedy(self):
        machine, disk, path = 0, 0, []
        while True:
            if len(self.log) == self.num:
                break
            condition = self.pre_count == 0
            if time.time() - self.start_time + self.time / self.count < 14.7:
                for task_id in np.where(condition & ((self.task_disk == disk) | (self.task_machine == machine) | (self.task_disk + self.task_machine == -2)))[0]:
                    self.update_task(task_id)
                task_id = masked_argmin(self.a + self.total_time, condition)
            else:
                task_id = masked_argmin(self.arrive_time + self.total_time, condition)
                self.update_task(task_id)
            self.use[self.task_disk[task_id]] -= self.data[task_id]
            self.machine_end_time[self.task_machine[task_id]] = self.d[task_id]
            machine, disk = self.task_machine[task_id], self.task_disk[task_id]
            self.log.append([task_id + 1, self.a[task_id], machine + 1, disk + 1])
            path.append(task_id)
            after_data_id = self.after_data[task_id]
            self.arrive_time[after_data_id] = np.maximum(self.arrive_time[after_data_id], self.d[task_id])
            self.r[after_data_id] = self.r[after_data_id] + self.d[task_id] - self.c[task_id]
            self.pre_count[after_data_id] -= 1
            after_task_id = self.after_task[task_id]
            self.arrive_time[after_task_id] = np.maximum(self.arrive_time[after_task_id], self.c[task_id])
            self.pre_count[after_task_id] -= 1
            self.pre_count[task_id] = 1
        return path, self.d.max(), self.log

    def get_solution(self, path):
        t = time.time()
        self.reset()
        i = 0
        while True:
            if len(self.log) == self.num:
                break
            task_id = path[i]
            if self.pre_count[task_id]:
                task_id = masked_argmin(self.total_time, (self.pre_count == 0) & (self.arrive_time >= self.machine_end_time.max()))
            else:
                i += 1
            self.update_task(task_id)
            self.use[self.task_disk[task_id]] -= self.data[task_id]
            self.machine_end_time[self.task_machine[task_id]] = self.d[task_id]
            self.log.append([task_id + 1, self.a[task_id], self.task_machine[task_id] + 1, self.task_disk[task_id] + 1])
            after_data_id = self.after_data[task_id]
            self.arrive_time[after_data_id] = np.maximum(self.arrive_time[after_data_id], self.d[task_id])
            self.r[after_data_id] = self.r[after_data_id] + self.d[task_id] - self.c[task_id]
            self.pre_count[after_data_id] -= 1
            after_task_id = self.after_task[task_id]
            self.arrive_time[after_task_id] = np.maximum(self.arrive_time[after_task_id], self.c[task_id])
            self.pre_count[after_task_id] -= 1
            self.pre_count[task_id] = 1
        self.time = self.time + time.time() - t
        self.count += 1
        return path, self.d.max(), self.log

    def local_search(self, path, score, log):
        path = [0] + path + [0]
        n = len(path)
        for i in range(0, n - 1):
            for j in range(i + 1, n - 1):
                if time.time() - self.start_time + self.time / self.count > 14.7:
                    return path, score, log
                path_ = path[:i + 1] + path[j:i:-1] + path[j + 1:]
                path_, score_, log_ = self.get_solution(path_[1:-1])
                if score_ < score:
                    return path_, score_, log_
        for i in range(1, n - 1):
            for k in range(0, min(n - i - 2, 20)):
                for j in range(i + 1 + k, n):
                    if time.time() - self.start_time + self.time / self.count > 14.7:
                        return path, score, log
                    if (i == j) or (j == i - 1):
                        continue
                    if i < j:
                        path_ = path[:i] + path[i + 1 + k:j] + path[i:i + 1 + k] + path[j:]
                    else:
                        path_ = path[:i] + path[i + 1 + k:j] + path[i:i + 1 + k] + path[j:]
                    path_, score_, log_ = self.get_solution(path_[1:-1])
                    if score_ < score:
                        return path_, score_, log_
        for i in range(1, n - 1):
            for k in range(0, min(n - i - 2, 20)):
                for j in range(i + k + 1, n - k - 1):
                    if time.time() - self.start_time + self.time / self.count > 14.7:
                        return path, score, log
                    if i == j:
                        continue
                    path_ = path[:i] + path[j:j + k + 1] + path[i + k + 1:j] + path[i:i + k + 1] + path[j + k + 1:]
                    path_, score_, log_ = self.get_solution(path_[1:-1])
                    if score_ < score:
                        return path_, score_, log_
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                if time.time() - self.start_time + self.time / self.count > 14.7:
                    return path, score, log
                path_ = [path[0]] + [path[i]] + path[1:i] + path[i + 1:j] + path[j + 1:-1] + [path[j]] + [path[-1]]
                path_, score_, log_ = self.get_solution(path_[1:-1])
                if score_ < score:
                    return path_, score_, log_
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                if time.time() - self.start_time + self.time / self.count > 14.7:
                    return path, score, log
                path_ = [path[0]] + [path[j]] + path[1:i] + path[i + 1:j] + path[j + 1:-1] + [path[i]] + [path[-1]]
                path_, score_, log_ = self.get_solution(path_[1:-1])
                if score_ < score:
                    return path_, score_, log_
        return path, score, log

    def optimize(self):
        if self.num > 100:
            path, score, log = self.get_solution(list(np.argsort(self.total_time)))
        else:
            path, score, log = self.solve_greedy()
        patient = 0
        while True:
            if time.time() - self.start_time + self.time / self.count > 14.7:
                return log
            if patient == 10:
                return log
            path_, score_, result_ = self.local_search(path, score, log)
            if score == score_:
                patient += 1
            else:
                patient = 0
            path, score, log = path_, score_, result_


def run():
    import platform
    if platform.system() == 'Windows':
        from io import StringIO
        with open('测试/1000.txt', 'r') as f:
            test_input = ''.join(f.readlines())
            sys.stdin = StringIO(test_input)
    env = JobSchedulingDataAssignmentProblem()
    log = env.optimize()
    result = "\n".join(map(lambda x: f'{x[0]} {x[1]} {x[2]} {x[3]}', log))
    sys.stdout.write(result)


if __name__ == '__main__':
    run()
