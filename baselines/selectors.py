from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def run_baselines(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    task: str,
    scoring: str,
    cv: int,
    n_select: int,
) -> List[Dict[str, Any]]:
    """
    تشغيل عدة طرق اختيار ميزات تقليدية لإعطاء مقارنة مع GA.
    يعيد قائمة من القواميس مع المفاتيح: name, selected_features, n_features, score.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2, RFE
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    results: List[Dict[str, Any]] = []
    n_features_total = X.shape[1]
    k = max(1, min(n_select, n_features_total))

    # Helper to evaluate a support mask
    def eval_mask(mask: List[int], name: str) -> None:
        idx = [i for i, b in enumerate(mask) if b]
        if len(idx) == 0:
            return
        Xs = X[:, idx]
        if task == 'classification':
            est = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, solver='liblinear')),
            ])
        else:
            est = Pipeline([
                ('scaler', StandardScaler()),
                ('reg', Ridge(alpha=1.0)),
            ])
        try:
            score = float(np.mean(cross_val_score(est, Xs, y, scoring=scoring, cv=cv)))
        except Exception:
            score = float('nan')
        results.append({
            'name': name,
            'selected_features': [feature_names[i] for i in idx],
            'n_features': len(idx),
            'score': score,
        })

    # ANOVA / F-test
    try:
        if task == 'classification':
            skb = SelectKBest(score_func=f_classif, k=k)
        else:
            skb = SelectKBest(score_func=f_regression, k=k)
        skb.fit(X, y)
        mask = list(skb.get_support())
        eval_mask(mask, 'SelectKBest-F')
    except Exception:
        pass

    # Chi2 (يحتاج قيم غير سالبة)
    try:
        scaler = MinMaxScaler()
        X_pos = scaler.fit_transform(X)
        if task == 'classification':
            skb2 = SelectKBest(score_func=chi2, k=k)
            skb2.fit(X_pos, y)
            mask = list(skb2.get_support())
            eval_mask(mask, 'SelectKBest-Chi2')
    except Exception:
        pass

    # RFE
    try:
        if task == 'classification':
            base = LogisticRegression(max_iter=1000, solver='liblinear')
        else:
            base = Ridge(alpha=1.0)
        rfe = RFE(estimator=base, n_features_to_select=k)
        rfe.fit(X, y)
        mask = list(rfe.get_support())
        eval_mask(mask, 'RFE')
    except Exception:
        pass

    # SelectFromModel - L1 or RF
    try:
        if task == 'classification':
            l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
            from sklearn.feature_selection import SelectFromModel
            sfm = SelectFromModel(l1, max_features=k)
            sfm.fit(X, y)
            mask = list(sfm.get_support())
            # إذا عدد المختار أقل من k بشكل كبير، نجرّب RF
            if sum(mask) < max(1, k // 2):
                rf = RandomForestClassifier(n_estimators=200, random_state=0)
                sfm = SelectFromModel(rf, max_features=k)
                sfm.fit(X, y)
                mask = list(sfm.get_support())
            eval_mask(mask, 'SelectFromModel')
        else:
            from sklearn.feature_selection import SelectFromModel
            lasso = Lasso(alpha=0.001, max_iter=5000)
            sfm = SelectFromModel(lasso, max_features=k)
            sfm.fit(X, y)
            mask = list(sfm.get_support())
            if sum(mask) < max(1, k // 2):
                rf = RandomForestRegressor(n_estimators=200, random_state=0)
                sfm = SelectFromModel(rf, max_features=k)
                sfm.fit(X, y)
                mask = list(sfm.get_support())
            eval_mask(mask, 'SelectFromModel')
    except Exception:
        pass

    return results
