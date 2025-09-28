from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class GAResult:
    best_mask: List[int]
    best_score: float
    history: List[Dict[str, Any]]
    n_features: int


class GeneticFeatureSelector:
    """
    محدد ميزات بالخوارزمية الجينية.

    - تمثيل الكروموسوم: قناع ثنائي بطول عدد الميزات.
    - التقييم: متوسط الدرجات عبر Cross-Validation مع عقوبة على عدد الميزات.
    - الانتقاء: بطولة (Tournament Selection).
    - العبور: موحد (Uniform) باحتمال crossover_rate.
    - الطفرة: قلب البتات باحتمال mutation_rate لكل جين.

    ملاحظات:
    - إذا لم تقدّم "estimator" يتم اختيار نموذج افتراضي حسب نوع المهمة.
    - يتم تحديد المقياس تلقائياً عند اختيار 'auto'.
    - يدعم التوقف المبكر عند عدم تحسن الأفضل لفترة "patience".
    - يوجد تخزين مؤقت (Caching) لقيم اللياقة للأقنعة التي تم تقييمها.
    """

    def __init__(
        self,
        estimator: Optional[object] = None,
        population_size: int = 30,
        generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        elitism: int = 2,
        scoring: str = 'auto',
        cv: int = 5,
        feature_cost: float = 0.01,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
        patience: int = 8,
        tournament_k: int = 3,
    ) -> None:
        self.estimator = estimator
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.scoring = scoring
        self.cv = cv
        self.feature_cost = feature_cost
        self.max_features = max_features
        self.random_state = random_state
        self.patience = patience
        self.tournament_k = tournament_k

        # Outputs
        self.best_mask_: Optional[List[int]] = None
        self.best_score_: Optional[float] = None
        self.history_: List[Dict[str, Any]] = []
        self.n_features_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.task_: Optional[str] = None

        if random_state is not None:
            random.seed(random_state)

    # --------- Public API ---------
    def fit(
        self,
        X,
        y,
        feature_names: Optional[Sequence[str]] = None,
        task: str = 'auto',
    ) -> GAResult:
        from sklearn.model_selection import cross_val_score
        import numpy as np

        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        self.n_features_ = n_features
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]

        # Determine task/scoring/default estimator
        task = self._infer_task(y) if task == 'auto' else task
        self.task_ = task
        # auto-scoring with multiclass awareness
        unique = np.unique(y)
        if self.scoring == 'auto':
            if task == 'classification':
                scoring = 'roc_auc' if len(unique) == 2 else 'roc_auc_ovr'
            else:
                scoring = 'r2'
        else:
            scoring = self.scoring
        estimator = self.estimator or self._default_estimator(task)

        # Constraints
        max_feats = self.max_features if self.max_features is not None else n_features
        max_feats = max(1, min(max_feats, n_features))

        # Initialize population
        population = [self._random_mask(n_features, max_feats) for _ in range(self.population_size)]
        cache: Dict[Tuple[int, ...], float] = {}

        best_mask = None
        best_score = -math.inf
        no_improve = 0

        for gen in range(self.generations):
            # Evaluate population
            fitnesses = []
            for mask in population:
                key = tuple(mask)
                if key in cache:
                    fit = cache[key]
                else:
                    if sum(mask) == 0:
                        fit = -math.inf
                    else:
                        Xs = X[:, mask == 1] if isinstance(mask, np.ndarray) else X[:, [i for i, b in enumerate(mask) if b]]
                        try:
                            scores = cross_val_score(estimator, Xs, y, scoring=scoring, cv=self.cv, n_jobs=None)
                            mean_score = float(np.mean(scores))
                        except Exception:
                            # في حالة فشل التدريب لأي سبب (مثل عدد العينات القليل)، نعطي درجة منخفضة
                            mean_score = -1e9
                        penalty = self.feature_cost * (float(sum(mask)) / float(n_features))
                        fit = mean_score - penalty
                    cache[key] = fit
                fitnesses.append(fit)

            # Track best
            gen_best_idx = int(max(range(len(population)), key=lambda i: fitnesses[i]))
            gen_best = population[gen_best_idx]
            gen_best_score = fitnesses[gen_best_idx]
            gen_mean = float(np.mean(fitnesses))
            gen_std = float(np.std(fitnesses))

            self.history_.append({
                'generation': gen,
                'best_score': gen_best_score,
                'mean_score': gen_mean,
                'std_score': gen_std,
                'best_size': int(sum(gen_best)),
            })

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_mask = list(gen_best)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

            # Create next generation
            next_pop: List[List[int]] = []

            # Elitism
            elite_indices = list(sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True))[: self.elitism]
            for i in elite_indices:
                next_pop.append(list(population[i]))

            # Fill the rest
            while len(next_pop) < self.population_size:
                p1 = self._tournament_select(population, fitnesses, self.tournament_k)
                p2 = self._tournament_select(population, fitnesses, self.tournament_k)
                c1, c2 = self._crossover(p1, p2, self.crossover_rate)
                c1 = self._mutate(c1, self.mutation_rate)
                c2 = self._mutate(c2, self.mutation_rate)
                c1 = self._repair(c1, max_feats)
                c2 = self._repair(c2, max_feats)
                next_pop.extend([c1, c2])

            population = next_pop[: self.population_size]

        # Save results
        self.best_mask_ = best_mask if best_mask is not None else [1] + [0] * (n_features - 1)
        self.best_score_ = best_score
        return GAResult(
            best_mask=self.best_mask_,
            best_score=self.best_score_,
            history=self.history_,
            n_features=n_features,
        )

    def get_support(self) -> List[bool]:
        if self.best_mask_ is None:
            raise RuntimeError('Call fit() first')
        return [bool(x) for x in self.best_mask_]

    def transform(self, X):
        import numpy as np
        if self.best_mask_ is None:
            raise RuntimeError('Call fit() first')
        X = np.asarray(X)
        mask_idx = [i for i, b in enumerate(self.best_mask_) if b]
        return X[:, mask_idx]

    # --------- Internal helpers ---------
    def _infer_task(self, y) -> str:
        import numpy as np
        y = np.asarray(y)
        # تصنيف إذا كانت عدد القيم المميزة قليل أو نوع الهدف غير عددي
        unique = np.unique(y)
        if y.dtype.kind in {'U', 'S', 'O', 'b'}:
            return 'classification'
        if len(unique) <= 20:
            return 'classification'
        return 'regression'

    def _infer_scoring(self, task: str) -> str:
        # لم تعد مستخدمة للـ auto لأننا نحددها داخل fit مع معرفة تعدد الفئات
        if task == 'classification':
            return 'roc_auc'
        return 'r2'

    def _default_estimator(self, task: str):
        # نستخدم Pipeline مع StandardScaler للاستقرار
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        if task == 'classification':
            from sklearn.linear_model import LogisticRegression
            return Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, solver='liblinear')),
            ])
        else:
            from sklearn.linear_model import Ridge
            return Pipeline([
                ('scaler', StandardScaler()),
                ('reg', Ridge(alpha=1.0)),
            ])

    def _random_mask(self, n_features: int, max_feats: int) -> List[int]:
        # نضمن وجود ميزة واحدة على الأقل
        k = random.randint(1, max_feats)
        indices = random.sample(range(n_features), k)
        mask = [0] * n_features
        for i in indices:
            mask[i] = 1
        return mask

    def _tournament_select(self, population: List[List[int]], fitnesses: List[float], k: int) -> List[int]:
        contenders = random.sample(range(len(population)), k)
        best_i = max(contenders, key=lambda i: fitnesses[i])
        return list(population[best_i])

    def _crossover(self, p1: List[int], p2: List[int], rate: float) -> Tuple[List[int], List[int]]:
        if random.random() > rate:
            return list(p1), list(p2)
        # Uniform crossover
        c1, c2 = [], []
        for a, b in zip(p1, p2):
            if random.random() < 0.5:
                c1.append(a)
                c2.append(b)
            else:
                c1.append(b)
                c2.append(a)
        return c1, c2

    def _mutate(self, child: List[int], rate: float) -> List[int]:
        return [1 - g if random.random() < rate else g for g in child]

    def _repair(self, mask: List[int], max_feats: int) -> List[int]:
        # ضمان القيود: على الأقل 1 وعلى الأكثر max_feats
        active = [i for i, b in enumerate(mask) if b]
        if len(active) == 0:
            i = random.randrange(len(mask))
            mask[i] = 1
            return mask
        if len(active) > max_feats:
            to_zero = random.sample(active, len(active) - max_feats)
            for i in to_zero:
                mask[i] = 0
        return mask
