import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.special import gamma
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class DS_IHBO:
    """
    Dynamic Surrogate-assisted Improved Hybrid Breeding Optimization for Feature Selection
    """
    
    def __init__(self, n_pop=45, max_iter=100, lambda_weight=0.99, 
                 sigma_range=(0.1, 1.0), sc_range=(4, 10), k_range=(4, 8),
                 F=0.5, T_s=75, T_n=6):
        """
        Initialize DS-IHBO parameters
        
        Args:
            n_pop: Population size
            max_iter: Maximum iterations
            lambda_weight: Weight for fitness function (λ in paper)
            sigma_range: Range for t-distribution mutation scale
            sc_range: Range for selfing count
            k_range: Range for KNN neighbors
            F: Scaling factor for differential evolution
            T_s: Number of iterations for surrogate evaluation phase
            T_n: Surrogate unit update trigger threshold
        """
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.lambda_weight = lambda_weight
        self.sigma_min, self.sigma_max = sigma_range
        self.sc_min, self.sc_max = sc_range
        self.k_min, self.k_max = k_range
        self.F = F
        self.T_s = T_s
        self.T_n = T_n
        
        # For population division
        self.m = n_pop // 3
        
        # Surrogate units storage
        self.surrogate_units = []
        self.current_surrogate_idx = 0
        
    def symmetric_uncertainty(self, X, y):
        """
        Calculate symmetric uncertainty between features and target
        Equation (5) in the paper
        """
        def entropy(x):
            _, counts = np.unique(x, return_counts=True)
            probs = counts / len(x)
            return -np.sum(probs * np.log2(probs + 1e-10))
        
        def mutual_info(x, y):
            xy = np.column_stack([x, y])
            unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
            p_xy = counts_xy / len(xy)
            
            _, counts_x = np.unique(x, return_counts=True)
            p_x = counts_x / len(x)
            
            _, counts_y = np.unique(y, return_counts=True)
            p_y = counts_y / len(y)
            
            mi = 0
            for i, (xi, yi) in enumerate(unique_xy):
                px = p_x[np.unique(x, return_inverse=True)[1][np.where(x == xi)[0][0]]]
                py = p_y[np.unique(y, return_inverse=True)[1][np.where(y == yi)[0][0]]]
                mi += p_xy[i] * np.log2((p_xy[i] + 1e-10) / (px * py + 1e-10))
            
            return mi
        
        n_features = X.shape[1]
        su_values = np.zeros(n_features)
        
        for i in range(n_features):
            mi = mutual_info(X[:, i], y)
            h_x = entropy(X[:, i])
            h_y = entropy(y)
            su_values[i] = 2 * mi / (h_x + h_y + 1e-10)
        
        return su_values
    
    def calculate_relevance_redundancy_index(self, X, y):
        """
        Calculate relevance-redundancy index (ψ) for each feature
        Equation (6) and (7) in the paper
        """
        n_features = X.shape[1]
        psi = np.zeros(n_features)
        
        # Calculate mutual information between each feature and class
        for i in range(n_features):
            # Relevance: MI between feature i and class
            relevance = self.symmetric_uncertainty(X[:, [i]], y)[0]
            
            # Redundancy: average MI with other features
            redundancy = 0
            for j in range(n_features):
                if i != j:
                    redundancy += self.symmetric_uncertainty(X[:, [i]], X[:, [j]])[0]
            redundancy /= (n_features - 1) if n_features > 1 else 1
            
            psi[i] = relevance - redundancy
        
        # Min-max normalization
        psi_normalized = (psi - np.min(psi)) / (np.max(psi) - np.min(psi) + 1e-10)
        
        return psi_normalized
    
    def remove_irrelevant_features(self, X, y):
        """
        Remove irrelevant features based on threshold ρ₀
        Equation (8) in the paper
        """
        psi = self.calculate_relevance_redundancy_index(X, y)
        D = X.shape[1]
        
        # Calculate threshold ρ₀
        psi_max = np.max(psi)
        k = int(D / np.log(D)) - 1 if D > 1 else 0
        k = max(0, min(k, D - 1))
        psi_sorted = np.sort(psi)[::-1]
        rho_0 = min(0.1 * psi_max, psi_sorted[k])
        
        # Select features above threshold
        selected_features = np.where(psi >= rho_0)[0]
        
        return selected_features, psi
    
    def hierarchical_population_initialization(self, n_features, psi):
        """
        Initialize population with asymmetric feature stratification
        Algorithm 1 in the paper
        """
        population = np.zeros((self.n_pop, n_features), dtype=int)
        
        # Sort features by relevance-redundancy index
        sorted_indices = np.argsort(psi)[::-1]
        
        # Divide features into three groups
        high_rel_idx = sorted_indices[:int(0.1 * n_features)]
        mid_rel_idx = sorted_indices[int(0.1 * n_features):int(0.5 * n_features)]
        low_rel_idx = sorted_indices[int(0.5 * n_features):]
        
        for n in range(self.n_pop):
            # Random number of features to initialize
            mu = np.random.randint(0, int(0.3 * n_features) + 1)
            
            # Select all high-relevance features
            population[n, high_rel_idx] = 1
            
            # Randomly select mid and low relevance features
            theta = np.random.uniform(0.7, 1.0)
            n_mid = int(theta * mu)
            n_low = mu - n_mid
            
            if n_mid > 0 and len(mid_rel_idx) > 0:
                selected_mid = np.random.choice(mid_rel_idx, 
                                              min(n_mid, len(mid_rel_idx)), 
                                              replace=False)
                population[n, selected_mid] = 1
            
            if n_low > 0 and len(low_rel_idx) > 0:
                selected_low = np.random.choice(low_rel_idx, 
                                              min(n_low, len(low_rel_idx)), 
                                              replace=False)
                population[n, selected_low] = 1
        
        return population
    
    def adaptive_differential_operators(self, t):
        """
        Calculate adaptive probabilities for differential operators
        Equations (12)-(17) in the paper
        """
        s_min = 0.1
        s_max = 1.0
        peak = s_max - s_min
        
        # S1: for DE1 (global search)
        s1 = peak / (1 + np.exp(-(t - self.max_iter/6) / (self.max_iter/25))) + s_min
        
        # S2: for DE2 (transition)
        s2 = peak * np.exp(-((t - self.max_iter/2)**2) / (10 * self.max_iter)) + s_min
        
        # S3: for DE3 (local search)
        s3 = peak / (1 + np.exp(-(t - 5*self.max_iter/6) / (self.max_iter/25))) + s_min
        
        # Normalize probabilities
        S = s1 + s2 + s3
        p1 = s1 / S
        p2 = s2 / S
        p3 = s3 / S
        
        return [p1, p2, p3]
    
    def update_maintainer_line(self, maintainer, population, best_idx, t):
        """
        Update maintainer line using adaptive differential operators
        Equations (9)-(11) in the paper
        """
        probs = self.adaptive_differential_operators(t)
        new_maintainer = maintainer.copy()
        
        for i in range(len(maintainer)):
            # Select differential operator based on probabilities
            op_choice = np.random.choice([0, 1, 2], p=probs)
            
            # Get random individuals
            indices = np.random.choice(len(population), 4, replace=False)
            
            if op_choice == 0:  # DE1: Global search
                r1 = np.random.random()
                new_maintainer[i] = maintainer[i] + r1 * (population[indices[0]] - population[indices[1]]) + \
                                   (1 - r1) * (population[indices[2]] - maintainer[i])
            
            elif op_choice == 1:  # DE2: Transition
                new_maintainer[i] = maintainer[i] + self.F * (population[indices[0]] - population[indices[1]]) + \
                                   self.F * (population[best_idx] - maintainer[i])
            
            else:  # DE3: Local search
                r2 = np.random.random()
                new_maintainer[i] = population[best_idx] + r2 * (population[indices[0]] - population[indices[1]]) + \
                                   (1 - r2) * (population[indices[2]] - population[indices[3]])
            
            # Binary conversion
            new_maintainer[i] = np.where(np.random.random(len(new_maintainer[i])) < 
                                        1 / (1 + np.exp(-new_maintainer[i])), 1, 0)
        
        return new_maintainer
    
    def t_distribution_mutation(self, t):
        """
        Generate t-distribution mutation for hybridization
        Equations (19)-(21) in the paper
        """
        # Calculate degree of freedom
        df = 2 + (t / self.max_iter) * 28
        
        # Calculate mutation scale
        gr = 2  # Growth rate
        sigma = (self.sigma_max - self.sigma_min) * (1 - (t / self.max_iter) ** gr) + self.sigma_min
        
        # Generate random number from t-distribution
        return t_dist.rvs(df, scale=sigma)
    
    def hybridization(self, maintainer, sterile, t):
        """
        Hybridization operation with t-distribution mutation
        Equation (18) in the paper
        """
        new_sterile = np.zeros_like(sterile)
        
        for i in range(len(sterile)):
            # Random selection from maintainer and sterile lines
            m_idx = np.random.randint(len(maintainer))
            s_idx = np.random.randint(len(sterile))
            
            # t-distribution mutation
            t_val = self.t_distribution_mutation(t)
            t_val = max(0, min(1, t_val))  # Clip to [0, 1]
            
            new_sterile[i] = t_val * sterile[s_idx] + (1 - t_val) * maintainer[m_idx]
            
            # Binary conversion
            new_sterile[i] = np.where(np.random.random(len(new_sterile[i])) < 
                                     1 / (1 + np.exp(-new_sterile[i])), 1, 0)
        
        return new_sterile
    
    def selfing(self, restorer, best_individual):
        """
        Selfing operation
        Equation (2) in the paper
        """
        new_restorer = np.zeros_like(restorer)
        
        for i in range(len(restorer)):
            j = np.random.choice([idx for idx in range(len(restorer)) if idx != i])
            r3 = np.random.random()
            
            new_restorer[i] = r3 * (best_individual - restorer[j]) + restorer[i]
            
            # Binary conversion
            new_restorer[i] = np.where(np.random.random(len(new_restorer[i])) < 
                                      1 / (1 + np.exp(-new_restorer[i])), 1, 0)
        
        return new_restorer
    
    def calculate_fitness(self, population, X, y, use_cv=True):
        """
        Calculate fitness values for population
        Equation (25) in the paper
        """
        fitness = np.zeros(len(population))
        
        for i, individual in enumerate(population):
            selected_features = np.where(individual == 1)[0]
            
            if len(selected_features) == 0:
                fitness[i] = 1.0  # Worst fitness
                continue
            
            X_selected = X[:, selected_features]
            
            if use_cv:
                # 5-fold cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                f1_scores = []
                
                for train_idx, val_idx in cv.split(X_selected, y):
                    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    clf = SVC(C=4, random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_val)
                    
                    f1_scores.append(f1_score(y_val, y_pred, average='weighted'))
                
                error = 1 - np.mean(f1_scores)
            else:
                # Direct evaluation
                clf = SVC(C=4, random_state=42)
                clf.fit(X_selected, y)
                y_pred = clf.predict(X_selected)
                error = 1 - f1_score(y, y_pred, average='weighted')
            
            # Feature ratio
            d = len(selected_features)
            D = X.shape[1]
            
            # Fitness function
            fitness[i] = self.lambda_weight * error + (1 - self.lambda_weight) * (d / D)
        
        return fitness
    
    def create_surrogate_units(self, X, y):
        """
        Create surrogate units using adaptive KNN and clustering
        Section 3.4(i) in the paper
        """
        n_samples = len(X)
        
        # Adaptive KNN for boundary detection
        k = int(np.sqrt(n_samples))
        k = max(self.k_min, min(self.k_max, k))
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Calculate local density
        local_density = np.mean(distances[:, 1:], axis=1)
        density_threshold = np.median(local_density)
        
        # Determine boundary samples
        is_boundary = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            k_i = self.k_max if local_density[i] <= density_threshold else self.k_min
            neighbors = indices[i, 1:k_i+1]
            neighbor_labels = y[neighbors]
            is_boundary[i] = not np.all(neighbor_labels == y[i])
        
        # Separate majority and minority classes
        unique_classes, class_counts = np.unique(y, return_counts=True)
        avg_count = np.mean(class_counts)
        
        majority_classes = unique_classes[class_counts > avg_count]
        minority_classes = unique_classes[class_counts <= avg_count]
        
        # Process samples
        representative_samples = []
        representative_labels = []
        
        # Process majority classes
        for cls in majority_classes:
            cls_mask = y == cls
            cls_boundary = is_boundary & cls_mask
            cls_central = ~is_boundary & cls_mask
            
            # Keep all boundary samples
            boundary_idx = np.where(cls_boundary)[0]
            representative_samples.extend(X[boundary_idx])
            representative_labels.extend(y[boundary_idx])
            
            # Randomly select 40% of central samples
            central_idx = np.where(cls_central)[0]
            if len(central_idx) > 0:
                n_select = max(1, int(0.4 * len(central_idx)))
                selected_idx = np.random.choice(central_idx, n_select, replace=False)
                representative_samples.extend(X[selected_idx])
                representative_labels.extend(y[selected_idx])
        
        # Process minority classes with SMOTE
        if len(minority_classes) > 0 and len(majority_classes) > 0:
            target_count = int(np.mean([np.sum(y == cls) for cls in majority_classes]))
            
            for cls in minority_classes:
                cls_mask = y == cls
                cls_samples = X[cls_mask]
                cls_labels = y[cls_mask]
                
                if len(cls_samples) < target_count and len(cls_samples) > 1:
                    # Apply SMOTE
                    smote = SMOTE(sampling_strategy={cls: target_count}, 
                                 k_neighbors=min(len(cls_samples)-1, 5),
                                 random_state=42)
                    
                    # Create temporary dataset for SMOTE
                    temp_X = np.vstack([cls_samples, X[~cls_mask][:min(100, len(X[~cls_mask]))]])
                    temp_y = np.hstack([cls_labels, y[~cls_mask][:min(100, len(y[~cls_mask]))]])
                    
                    try:
                        X_resampled, y_resampled = smote.fit_resample(temp_X, temp_y)
                        cls_resampled = X_resampled[y_resampled == cls]
                        
                        # Determine boundary for resampled
                        cls_boundary = is_boundary[cls_mask]
                        n_boundary = np.sum(cls_boundary)
                        n_central = len(cls_resampled) - n_boundary
                        
                        # Add samples
                        representative_samples.extend(cls_resampled[:n_boundary])
                        representative_labels.extend([cls] * n_boundary)
                        
                        if n_central > 0:
                            n_select = max(1, int(0.4 * n_central))
                            representative_samples.extend(cls_resampled[n_boundary:n_boundary+n_select])
                            representative_labels.extend([cls] * n_select)
                    except:
                        # If SMOTE fails, use original samples
                        representative_samples.extend(cls_samples)
                        representative_labels.extend(cls_labels)
                else:
                    representative_samples.extend(cls_samples)
                    representative_labels.extend(cls_labels)
        
        if len(representative_samples) == 0:
            return [(X, y)]  # Return original if no representatives
        
        representative_samples = np.array(representative_samples)
        representative_labels = np.array(representative_labels)
        
        # Create 6 surrogate units with different sizes
        surrogate_units = []
        proportions = [0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
        
        # Agglomerative clustering
        n_clusters = min(6, len(representative_samples))
        if n_clusters > 1:
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clustering.fit_predict(representative_samples)
        else:
            cluster_labels = np.zeros(len(representative_samples))
        
        for prop in proportions:
            target_size = int(prop * n_samples)
            selected_idx = []
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_samples = np.where(cluster_mask)[0]
                
                if len(cluster_samples) > 0:
                    n_select = min(len(cluster_samples), target_size // n_clusters)
                    selected = np.random.choice(cluster_samples, n_select, replace=False)
                    selected_idx.extend(selected)
            
            if len(selected_idx) > 0:
                surrogate_X = representative_samples[selected_idx]
                surrogate_y = representative_labels[selected_idx]
                surrogate_units.append((surrogate_X, surrogate_y))
        
        return surrogate_units if surrogate_units else [(X, y)]
    
    def select_best_surrogate(self, best_individual, original_X, original_y):
        """
        Select the most appropriate surrogate unit
        Section 3.4(ii) in the paper
        """
        # Calculate fitness on original dataset
        original_fitness = self.calculate_fitness([best_individual], original_X, original_y, use_cv=False)[0]
        
        # Calculate fitness on each surrogate unit
        min_error = float('inf')
        best_surrogate_idx = 0
        
        for i, (surrogate_X, surrogate_y) in enumerate(self.surrogate_units):
            surrogate_fitness = self.calculate_fitness([best_individual], surrogate_X, surrogate_y, use_cv=False)[0]
            error = abs(surrogate_fitness - original_fitness)
            
            if error < min_error:
                min_error = error
                best_surrogate_idx = i
        
        return best_surrogate_idx
    
    def fit(self, X, y):
        """
        Main DS-IHBO algorithm
        Algorithm 2 in the paper
        """
        # Remove irrelevant features
        selected_features, psi_full = self.remove_irrelevant_features(X, y)
        X = X[:, selected_features]
        psi = psi_full[selected_features]
        
        n_features = X.shape[1]
        
        # Create surrogate units
        self.surrogate_units = self.create_surrogate_units(X, y)
        
        # Initialize population
        population = self.hierarchical_population_initialization(n_features, psi)
        
        # Calculate initial fitness using original dataset
        fitness = self.calculate_fitness(population, X, y)
        
        # Find best individual
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Select initial surrogate unit
        self.current_surrogate_idx = self.select_best_surrogate(best_individual, X, y)
        
        # Tracking variables
        no_improvement_count = 0
        selfing_counts = np.zeros(self.m)
        
        for t in range(self.max_iter):
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Divide population
            maintainer = population[:self.m]
            restorer = population[self.m:2*self.m]
            sterile = population[2*self.m:]
            
            # Update maintainer line
            maintainer = self.update_maintainer_line(maintainer, population, 0, t)
            
            # Hybridization
            sterile = self.hybridization(maintainer, sterile, t)
            
            # Selfing with adaptive upper bound
            SC = self.sc_min + (self.sc_max - self.sc_min) * (1 - (t / self.max_iter) ** 2)
            
            for i in range(len(restorer)):
                if selfing_counts[i] < SC:
                    restorer[i] = self.selfing([restorer[i]], population[0])[0]
                    selfing_counts[i] += 1
                else:
                    # Renewal operation
                    restorer[i] = np.random.randint(0, 2, n_features)
                    selfing_counts[i] = 0
            
            # Combine population
            population = np.vstack([maintainer, restorer, sterile])
            
            # Evaluate fitness
            if t < self.T_s:
                # Use surrogate unit
                surrogate_X, surrogate_y = self.surrogate_units[self.current_surrogate_idx]
                fitness = self.calculate_fitness(population, surrogate_X, surrogate_y)
                
                # Evaluate best individual on original dataset
                best_idx = np.argmin(fitness)
                real_fitness = self.calculate_fitness([population[best_idx]], X, y)[0]
                
                # Check for improvement
                if real_fitness < best_fitness:
                    best_individual = population[best_idx].copy()
                    best_fitness = real_fitness
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Update surrogate if no improvement for T_n iterations
                if no_improvement_count >= self.T_n:
                    self.current_surrogate_idx = self.select_best_surrogate(best_individual, X, y)
                    no_improvement_count = 0
            else:
                # Use original dataset
                fitness = self.calculate_fitness(population, X, y)
                
                # Update best individual
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_fitness:
                    best_individual = population[best_idx].copy()
                    best_fitness = fitness[best_idx]
            
            # Print progress
            if (t + 1) % 10 == 0:
                n_selected = np.sum(best_individual)
                print(f"Iteration {t+1}/{self.max_iter}: Best fitness = {best_fitness:.4f}, "
                      f"Selected features = {n_selected}")
        
        # Final result
        self.best_individual = best_individual
        self.selected_features = selected_features[np.where(best_individual == 1)[0]]
        
        return self
    
    def transform(self, X):
        """
        Transform data using selected features
        """
        return X[:, self.selected_features]
    
    def fit_transform(self, X, y):
        """
        Fit and transform in one step
        """
        self.fit(X, y)
        return self.transform(X)


# Example usage
if __name__ == "__main__":
    # Generate sample imbalanced dataset
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    
    # Create an imbalanced high-dimensional dataset
    X, y = make_classification(n_samples=500, n_features=100, n_informative=20,
                              n_redundant=30, n_clusters_per_class=2,
                              weights=[0.9, 0.1], flip_y=0.05,
                              random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        stratify=y, random_state=42)
    
    # Initialize DS-IHBO
    print("Initializing DS-IHBO for feature selection...")
    ds_ihbo = DS_IHBO(n_pop=45, max_iter=100, T_s=75, T_n=6)
    
    # Perform feature selection
    print(f"Original features: {X_train.shape[1]}")
    print("Running DS-IHBO...")
    X_train_selected = ds_ihbo.fit_transform(X_train, y_train)
    X_test_selected = ds_ihbo.transform(X_test)
    
    print(f"Selected features: {len(ds_ihbo.selected_features)}")
    print(f"Selected feature indices: {ds_ihbo.selected_features}")
    
    # Evaluate on test set
    clf = SVC(C=4, random_state=42)
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
