import bisect
import copy
import random
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import t
from biomedical_selection.algorithm.base import BaseEA
from biomedical_selection.algorithm.init_helper import Chaos
from biomedical_selection.utils.helper import levy
from biomedical_selection.utils import FunFs

# def my_job(args):
#     binary_vector, fitness_func = args
#     fitness, acc, f1, gmean = fitness_func(binary_vector)
#     return fitness, acc, f1, gmean
def my_job(args):
    binary_vector, use_surrogate, fitness_func, surrogate_params = args
    if use_surrogate:
        fitness, acc, f1, gmean = surrogate_evaluate_static(binary_vector, *surrogate_params)
    else:
        fitness, acc, f1, gmean = fitness_func(binary_vector)
    return fitness, acc, f1, gmean

def surrogate_evaluate_static(binary_vector, X_train_surrogate, y_train_surrogate, X_val_surrogate, y_val_surrogate, surrogate_scaler, funfs_params):
    # 可选的特征缩放
    if surrogate_scaler is not None:
        X_train_scaled = surrogate_scaler.transform(X_train_surrogate)
        X_val_scaled = surrogate_scaler.transform(X_val_surrogate)
    else:
        X_train_scaled = X_train_surrogate
        X_val_scaled = X_val_surrogate

    # 提取被选择的特征
    selected_features = np.where(binary_vector == 1)[0]

    if len(selected_features) == 0:
        # 如果没有选择特征，返回最差的适应度值
        return float('inf'), 0, 0, 0

    X_train_selected = X_train_scaled[:, selected_features]
    X_val_selected = X_val_scaled[:, selected_features]

    # 后续代码保持不变，训练分类器并评估性能
    classifier_name = funfs_params.get('classifier_name', 'svm')
    clf_params = funfs_params.get('clf_params', {})

    # 训练分类器
    if classifier_name == 'svm':
        from sklearn.svm import SVC
        clf = SVC(**clf_params)
    elif classifier_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(**clf_params)
    elif classifier_name == 'xgboost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(**clf_params)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    clf.fit(X_train_selected, y_train_surrogate)

    # 在验证集上评估
    y_pred = clf.predict(X_val_selected)

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(y_val_surrogate, y_pred)
    f1 = f1_score(y_val_surrogate, y_pred, average='weighted')

    # 计算 G-Mean
    cm = confusion_matrix(y_val_surrogate, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        gmean = np.sqrt(sensitivity * specificity)
    else:
        # Calculate G-mean for multi-class
        n_classes = cm.shape[0]
        sensitivities = []
        specificities = []
        for i in range(n_classes):
            # Sensitivity (TPR) for class i
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivities.append(sensitivity)

            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)
        # Compute G-mean as the geometric mean of all classes' sensitivities and specificities
        # Only compute if there is no zero sensitivity or specificity
        if all(s > 0 for s in sensitivities) and all(s > 0 for s in specificities):
            gmean = np.sqrt(np.prod(sensitivities) * np.prod(specificities))
        else:
            gmean = 0  # Handle cases where any class has 0 sensitivity or specificity

    # 计算适应度函数值
    total_features = binary_vector.shape[0]
    num_selected_features = len(selected_features)
    fitness = 0.99 * (1 - acc) + 0.01 * (num_selected_features / total_features)
    return fitness, acc, f1, gmean

def generate_different_indices(i, arange, dim, size=1):
    possible_values = arange[arange != i]

    random_indices = np.random.choice(possible_values, size=(size, dim), replace=True)

    return random_indices if size != 1 else random_indices.reshape(dim)


class MSIHRO(BaseEA):
    def __init__(self, _np, n, uppers, lowers, surrogate_sets=None, funfs_params=None,**kwargs):
        BaseEA.__init__(self, _np, n, uppers, lowers, **kwargs)
        self.max_prob = 0.92
        self.min_prob = 0.01
        self.initial_scale = 1.0
        self.final_scale = 0.1
        self.growth_rate = 2
        self.F = 0.5
        self.cr = 0.9
        self.sc_max = 10
        self.sc_min = 4
        self.Is = 70  # 动态代理过程的迭代次数
        self.NI_step = 5  # 未改进的最大迭代次数
        self.no_improve_counter = 0  # 未改进计数器

        chaos = Chaos(_np, n, lowers, uppers, solution_class=kwargs.get("solution_class"))
        self.solutions = chaos.generate_population(map_funciton=kwargs.get("map_function", "good_point_set"))
        self.group_size = int(self.np / 3)

        # 加载代理集
        self.surrogate_sets = surrogate_sets if surrogate_sets is not None else []
        self.current_surrogate_set = None  # 当前使用的代理集
        self.use_surrogate = True  # 是否使用代理集进行评估
        # 初始化 FunFs 参数
        self.funfs_params = funfs_params if funfs_params is not None else {}

    def fit(self, gen):
        obl_solutions = [self.create_solution(True) for _ in range(self.np)]
        random_array = np.random.random((self.np, self.n))
        for i, ind in enumerate(self.solutions):
            obl_solutions[i].vector = random_array[i, :] * (self.uppers + self.lowers) - ind.vector
            out_of_bounds = (obl_solutions[i].vector < self.lowers) | (obl_solutions[i].vector > self.uppers)
            random_values = np.random.uniform(self.lowers, self.uppers, size=self.n)
            obl_solutions[i].vector = np.where(out_of_bounds, random_values, obl_solutions[i].vector)

        self.solutions += obl_solutions

        for ind in self.solutions:
            self.exchange_binary(ind)

        with ProcessPoolExecutor(self.nw) as pool:
            args = [(ind.binaryVector, False, self.fitness_func, None) for ind in self.solutions]
            results = pool.map(my_job, args)
            for ind, (fitness, acc, f1, gmean) in zip(self.solutions, results):
                ind.fitness, ind.acc, ind.f1, ind.gmean= fitness, acc, f1, gmean

            self.solutions.sort(key=lambda s: s.fitness)
            self.solutions = self.solutions[:self.np]
            self.best_solution = copy.deepcopy(self.solutions[0])
            # 使用 xbest 寻找最适合的代理集
            xbest = self.best_solution
            f0 = xbest.fitness
            surrogate_errors = []
            for surrogate_set in self.surrogate_sets:
                X_train_surrogate, y_train_surrogate, X_val_surrogate, y_val_surrogate, surrogate_scaler = surrogate_set
                fitness, _, _, _ = surrogate_evaluate_static(
                    xbest.binaryVector,
                    X_train_surrogate, y_train_surrogate,
                    X_val_surrogate, y_val_surrogate,
                    surrogate_scaler,
                    self.funfs_params)
                surrogate_errors.append(abs(fitness - f0))
            # 选择误差最小的代理集
            min_error_index = np.argmin(surrogate_errors)
            self.current_surrogate_set = self.surrogate_sets[min_error_index]

            maintainer_trial_solutions = [self.create_solution(True) for _ in range(self.group_size)]
            hybird_trial_solutions = [self.create_solution(True) for _ in range(2 * self.group_size, self.np)]
            selfing_trial_solutions = [self.create_solution(True) for _ in range(self.group_size)]

            start = time.perf_counter()
            real_fitness_prev = float('inf')  # 初始化为无穷大，确保第一轮迭代总是有改进
            for it in range(1, gen + 1):
                # 1. 决定是否使用代理模型
                if it <= self.Is:
                    self.use_surrogate = True
                    self.de_stage(maintainer_trial_solutions, it, gen, pool=pool)
                    self.hybridization_stage(hybird_trial_solutions, it, gen, pool=pool)
                    self.selfing_stage(selfing_trial_solutions, it, gen, pool=pool)
                    self.solutions.sort(key=lambda s: s.fitness)
                    if self.solutions[0].fitness < self.best_solution.fitness:
                        self.best_solution = copy.deepcopy(self.solutions[0])
                    # 使用原始训练集评估 g_best
                    real_fitness, real_acc, real_f1, real_gmean = self.fitness_func(self.solutions[0].binaryVector)
                    # 比较当前的 real_fitness 与上一轮的 real_fitness
                    if real_fitness < real_fitness_prev:
                        self.no_improve_counter = 0
                    else:
                        self.no_improve_counter += 1
                    # 更新上一轮的 real_fitness
                    real_fitness_prev = real_fitness
                    # 如果未改进计数器达到阈值，更新代理集
                    if self.no_improve_counter >= self.NI_step:
                        gbest = self.solutions[0]
                        surrogate_errors = []
                        for surrogate_set in self.surrogate_sets:
                            X_train_surrogate, y_train_surrogate, X_val_surrogate, y_val_surrogate, surrogate_scaler = surrogate_set
                            fitness, _, _, _ = surrogate_evaluate_static(
                                gbest.binaryVector,
                                X_train_surrogate, y_train_surrogate,
                                X_val_surrogate, y_val_surrogate,
                                surrogate_scaler,
                                self.funfs_params)
                            surrogate_errors.append(abs(fitness - real_fitness))  #代理-真实
                        if surrogate_errors:
                            min_error_index = np.argmin(surrogate_errors)
                            self.current_surrogate_set = self.surrogate_sets[min_error_index]
                        else:
                            self.current_surrogate_set = None
                        self.no_improve_counter = 0
                    self.add_best(self.best_solution)
                else:
                    self.use_surrogate = False
                    self.de_stage(maintainer_trial_solutions, it, gen, pool=pool)
                    self.hybridization_stage(hybird_trial_solutions, it, gen, pool=pool)
                    self.selfing_stage(selfing_trial_solutions, it, gen, pool=pool)
                    self.solutions.sort(key=lambda s: s.fitness)
                    # 3. 更新最佳解
                    if self.solutions[0].fitness < self.best_solution.fitness:
                        self.best_solution = copy.deepcopy(self.solutions[0])
                    self.add_best(self.best_solution)

                if it % 10 == 0:
                    print(
                        f"Iteration: {it}  |  Fitness: {self.best_solution.fitness: .3f}  |  Acc: {self.best_solution.acc * 100: .3f}  |  F1: {self.best_solution.f1 * 100: .3f}  | G-mean: {self.best_solution.gmean * 100: .3f}  |  Num: {sum(self.best_solution.binaryVector)}")
            end = time.perf_counter()
            self.optimization_time = end - start

    def de_stage(self, trial_solutions, cur_iteration, max_iteration, pool: ProcessPoolExecutor):
        arange = np.arange(0, self.group_size)

        probs = self.generate_probabilities(cur_iteration, max_iteration)
        de_strategy = self.roulette_wheel_selection(probs)
        if de_strategy == 0:
            # global
            for i, ind in enumerate(self.solutions[:self.group_size]):
                p1, p2, p3 = np.random.choice(arange[arange != i], size=3, replace=False)
                r1 = np.random.random(self.n)
                trial_solutions[i].vector = ind.vector + r1 * (
                        self.solutions[p1].vector - self.solutions[p2].vector) + (1 - r1) * (
                                                    self.solutions[p3].vector - ind.vector)
        elif de_strategy == 1:
            # transition period
            for i, ind in enumerate(self.solutions[:self.group_size]):
                p1, p2 = np.random.choice(arange[arange != i], size=2, replace=False)
                trial_solutions[i].vector = ind.vector + self.F * (
                        self.solutions[p1].vector - self.solutions[p2].vector) + self.F * (
                                                    self.best_solution.vector - ind.vector)
        else:
            # local
            for i, ind in enumerate(self.solutions[:self.group_size]):
                p1, p2, p3, p4 = np.random.choice(arange[arange != i], size=4, replace=False)
                r2 = np.random.random(self.n)
                trial_solutions[i].vector = self.best_solution.vector + r2 * (
                        self.solutions[p1].vector - self.solutions[p2].vector) + (1 - r2) * (
                                                    self.solutions[p3].vector - self.solutions[p4].vector)

        for i in range(self.group_size):
            idx = np.where(np.random.random(self.n) > self.cr)[0]
            trial_solutions[i].vector[idx] = self.solutions[i].vector[idx]
            self.exchange_binary(trial_solutions[i])

        # 使用 evaluate_solutions 进行适应度评估
        self.evaluate_solutions(trial_solutions, pool)

        # 更新解集
        for i, ind in enumerate(trial_solutions):
            if ind.fitness < self.solutions[i].fitness:
                self.solutions[i] = copy.deepcopy(ind)
                if ind.fitness < self.best_solution.fitness:
                    self.best_solution = copy.deepcopy(ind)

    def hybridization_stage(self, trial_solutions, cur_iteration, max_iteration, pool: ProcessPoolExecutor):
        shift = 2 * self.group_size
        arange = np.arange(shift, self.np)

        df = 2 + (cur_iteration / max_iteration) * 28
        scale = (self.initial_scale - self.final_scale) * (
                1 - (cur_iteration / max_iteration) ** self.growth_rate) + self.final_scale
        t_rand = t.rvs(df, scale, size=(self.np - shift, self.n))
        v = np.zeros_like(t_rand)

        maintainer_index = np.random.randint(0, self.group_size, (self.np - shift, self.n))
        sterile_index = np.array([generate_different_indices(i, arange, self.n) for i in range(shift, self.np)])
        for i in range(shift, self.np):
            z = i - shift
            trial_solutions[z].trial = self.solutions[i].trial

            v1 = np.asarray([self.solutions[k].vector[d] for d, k in enumerate(maintainer_index[z])])
            v2 = np.asarray([self.solutions[k].vector[d] for d, k in enumerate(sterile_index[z])])

            v[z] = t_rand[z] * v2 + (1 - t_rand[z]) * v1

            trial_solutions[z].vector = v[z, :]
            self.exchange_binary(trial_solutions[z])

        # 使用 evaluate_solutions 进行适应度评估
        self.evaluate_solutions(trial_solutions, pool)

        # 更新解集
        for i, ind in enumerate(trial_solutions):
            if ind.fitness < self.solutions[i + shift].fitness:
                self.solutions[i + shift] = copy.deepcopy(ind)
                if ind.fitness < self.best_solution.fitness:
                    self.best_solution = copy.deepcopy(ind)

    def selfing_stage(self, trial_solutions, cur_iteration, max_iteration, pool: ProcessPoolExecutor):
        c1 = 2 * np.exp(-(4 * cur_iteration / max_iteration) ** 2)
        max_trial = self.sc_min + (self.sc_max - self.sc_min) * (1 - (cur_iteration / max_iteration) ** 2)
        shift = self.group_size
        no_exceed_list = []
        exceed_list = []
        for i in range(self.group_size, 2 * self.group_size):
            z = i - shift
            if self.solutions[i].trial < max_trial:
                while True:
                    rd_idx = np.random.randint(self.group_size, 2 * self.group_size)
                    if rd_idx != i:
                        restorer_index = rd_idx
                        break

                trial_solutions[z].vector = np.random.random() * (
                        self.best_solution.vector - self.solutions[restorer_index].vector) + c1 * (
                                                    self.lowers + 0.1 * levy(self.n) * (self.uppers - self.lowers))

                self.exchange_binary(trial_solutions[z])
                no_exceed_list.append(z)
            else:
                r3 = random.random()
                trial_solutions[z].vector = r3 * (self.uppers - self.lowers) + self.solutions[i].vector + self.lowers

                self.exchange_binary(trial_solutions[z])
                exceed_list.append(z)

        # 使用 evaluate_solutions 进行适应度评估
        self.evaluate_solutions(trial_solutions, pool)

        # 更新解集
        for idx in no_exceed_list:
            z = idx + shift
            ind = trial_solutions[idx]
            if ind.fitness < self.solutions[z].fitness:
                self.solutions[z] = copy.deepcopy(ind)
                self.solutions[z].trial = 0
                if ind.fitness < self.best_solution.fitness:
                    self.best_solution = copy.deepcopy(ind)
            else:
                self.solutions[z].trial_increase()

        for idx in exceed_list:
            z = idx + shift
            ind = trial_solutions[idx]
            self.solutions[z] = copy.deepcopy(ind)
            self.solutions[z].trial = 0
            if ind.fitness < self.best_solution.fitness:
                self.best_solution = copy.deepcopy(ind)

    def generate_probabilities(self, it, T):
        peak = self.max_prob - self.min_prob
        s1 = peak / (1 + np.exp((it - T / 6) / (T / 25))) + self.min_prob
        s2 = peak * np.exp(-((it - T / 2) ** 2) / (T * 10)) + self.min_prob
        s3 = peak / (1 + np.exp(-(it - 5 * T / 6) / (T / 25))) + self.min_prob
        total = s1 + s2 + s3
        return [s1 / total, s2 / total, s3 / total]

    def roulette_wheel_selection(self, probs):
        cumsum = np.cumsum(probs)
        r = np.random.random()
        idx = bisect.bisect_left(cumsum, r)
        return idx

    def surrogate_evaluate(self, binary_vector, X_surrogate, y_surrogate):
        # 使用 FunFs 类进行评估
        funfs = FunFs(
            trainIn=X_surrogate,
            trainOut=y_surrogate,
            testSize=self.funfs_params.get('testSize', 0.2),
            classifier_name=self.funfs_params.get('classifier_name', 'svm'),
            scalers=self.funfs_params.get('scalers', None),
            clf_params=self.funfs_params.get('clf_params', {})
        )

        fitness, acc, f1, gmean = funfs.calculate_fitness(binary_vector)
        return fitness, acc, f1, gmean

    def evaluate_solutions(self, trial_solutions, pool):
        if self.use_surrogate:
            X_train_surrogate, y_train_surrogate, X_val_surrogate, y_val_surrogate, surrogate_scaler = self.current_surrogate_set
            surrogate_params = (
            X_train_surrogate, y_train_surrogate, X_val_surrogate, y_val_surrogate, surrogate_scaler, self.funfs_params)
            args = [(ind.binaryVector, True, None, surrogate_params) for ind in trial_solutions]
        else:
            args = [(ind.binaryVector, False, self.fitness_func, None) for ind in trial_solutions]

        results = pool.map(my_job, args)

        for ind, (fitness, acc, f1, gmean) in zip(trial_solutions, results):
            ind.fitness, ind.acc, ind.f1, ind.gmean = fitness, acc, f1, gmean



