from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def load_csv(
    file_path: str,
    target_column: str,
    task: str = 'auto',
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    تحميل ملف CSV وتجهيز X وy.
    - يتم اختيار الأعمدة الرقمية فقط كميزات في هذه النسخة.
    - يتم تعويض القيم المفقودة بالوسيط.
    - يتم ترميز الهدف إذا كان فئوياً.
    - يتم تحديد نوع المسألة تلقائياً إذا كانت 'auto'.
    """
    df = pd.read_csv(file_path)
    if target_column not in df.columns:
        raise ValueError(f"العمود المستهدف غير موجود: {target_column}")

    y_series = df[target_column]
    X_df = df.drop(columns=[target_column])

    # نحتفظ بالأعمدة الرقمية فقط في هذه المرحلة
    X_num = X_df.select_dtypes(include=['number']).copy()
    feature_names: List[str] = list(X_num.columns)

    if X_num.empty:
        raise ValueError("لا توجد ميزات رقمية في الملف. يرجى توفير أعمدة رقمية أو تحديث المعالجة لدعم الفئيات.")

    # إسقاط الصفوف التي يفتقد فيها الهدف
    mask_valid_y = ~y_series.isna()
    X_num = X_num.loc[mask_valid_y]
    y_series = y_series.loc[mask_valid_y]

    # تعويض القيم المفقودة بالوسيط
    imputer = SimpleImputer(strategy='median')
    X_proc = imputer.fit_transform(X_num.values)

    # تحديد نوع المسألة
    inferred_task = task
    if task == 'auto':
        if y_series.dtype.kind in {'U', 'S', 'O', 'b'}:
            inferred_task = 'classification'
        else:
            n_unique = y_series.nunique(dropna=True)
            inferred_task = 'classification' if n_unique <= 20 else 'regression'

    # ترميز الهدف إذا كان تصنيفاً وغير عددي
    y = y_series.values
    if inferred_task == 'classification' and y_series.dtype.kind in {'U', 'S', 'O'}:
        le = LabelEncoder()
        y = le.fit_transform(y_series.astype(str).values)
    else:
        y = y.astype(float)

    return X_proc, y, feature_names, inferred_task
