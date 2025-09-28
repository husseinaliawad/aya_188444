import os
import time
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from .forms import UploadDatasetForm
import json
from urllib.parse import quote
from django.urls import reverse
from engine.gafs import GeneticFeatureSelector
from utils.preprocessing import load_csv
from baselines.selectors import run_baselines


def index(request):
    return render(request, 'selector/index.html')


def upload_dataset(request):
    if request.method == 'POST':
        form = UploadDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                f = form.cleaned_data['file']
                target_column = form.cleaned_data['target_column']
                task = form.cleaned_data['task']
                population_size = form.cleaned_data['population_size']
                generations = form.cleaned_data['generations']
                crossover_rate = form.cleaned_data['crossover_rate']
                mutation_rate = form.cleaned_data['mutation_rate']
                elitism = form.cleaned_data['elitism']
                max_features = form.cleaned_data['max_features']
                scoring_choice = form.cleaned_data['scoring']
                cv = form.cleaned_data['cv']

                upload_dir = settings.MEDIA_ROOT / 'uploads'
                results_dir = settings.MEDIA_ROOT / 'results'
                os.makedirs(upload_dir, exist_ok=True)
                os.makedirs(results_dir, exist_ok=True)
                ts = int(time.time())
                safe_name = f"{ts}_" + os.path.basename(f.name)
                dest_path = upload_dir / safe_name
                with open(dest_path, 'wb+') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)

                # Load and preprocess
                X, y, feature_names, inferred_task = load_csv(str(dest_path), target_column, task)

                # Determine scoring
                if scoring_choice == 'auto':
                    import numpy as np
                    unique = np.unique(y)
                    if inferred_task == 'classification':
                        scoring = 'roc_auc' if len(unique) == 2 else 'roc_auc_ovr'
                    else:
                        scoring = 'r2'
                else:
                    scoring = scoring_choice

                # Run GA
                ga = GeneticFeatureSelector(
                    estimator=None,
                    population_size=population_size,
                    generations=generations,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    elitism=elitism,
                    scoring=scoring,
                    cv=cv,
                    feature_cost=0.02,
                    max_features=max_features,
                    random_state=42,
                    patience=max(5, min(10, generations // 2)),
                )
                ga_result = ga.fit(X, y, feature_names=feature_names, task=inferred_task)

                best_mask = ga_result.best_mask
                selected_idx = [i for i, b in enumerate(best_mask) if b]
                selected_features = [feature_names[i] for i in selected_idx]

                # Raw GA score (without penalty)
                from sklearn.model_selection import cross_val_score
                import numpy as np
                est = ga._default_estimator(inferred_task)
                Xs = X[:, selected_idx] if len(selected_idx) > 0 else X
                try:
                    ga_raw_score = float(np.mean(cross_val_score(est, Xs, y, scoring=scoring, cv=cv)))
                except Exception:
                    ga_raw_score = float('nan')

                # Baselines with same number of features
                n_select = len(selected_idx) if len(selected_idx) > 0 else (max_features or min(5, X.shape[1]))
                baselines = run_baselines(X, y, feature_names, inferred_task, scoring, cv, n_select)

                # Compose results payload
                import pandas as pd
                try:
                    df_head = pd.read_csv(str(dest_path), nrows=3).to_dict(orient='records')
                except Exception:
                    df_head = []
                results_payload = {
                    'dataset': {
                        'file': safe_name,
                        'path': str(dest_path),
                        'target_column': target_column,
                        'n_rows': int(len(y)),
                        'n_features_total': int(X.shape[1]),
                        'preview_head': df_head,
                    },
                    'task': inferred_task,
                    'scoring': scoring,
                    'cv': int(cv),
                    'ga': {
                        'selected_features': selected_features,
                        'n_selected': len(selected_features),
                        'score_raw': ga_raw_score,
                        'score_penalized': ga_result.best_score,
                        'history': ga_result.history,
                    },
                    'baselines': baselines,
                }

                results_name = f"{ts}_{os.path.splitext(os.path.basename(f.name))[0]}.json"
                results_path = results_dir / results_name
                with open(results_path, 'w', encoding='utf-8') as fp:
                    json.dump(results_payload, fp, ensure_ascii=False, indent=2)

                messages.success(request, 'تم تنفيذ الاختيار بنجاح! إليك النتائج.')
                return redirect(f"{reverse('selector:results')}?file={quote(results_name)}")
            except Exception as e:
                messages.error(request, f'حدث خطأ أثناء المعالجة: {e}')
                return redirect('selector:upload')
        else:
            messages.error(request, 'تحقق من الحقول المدخلة.')
    else:
        form = UploadDatasetForm()
    return render(request, 'selector/upload.html', {'form': form})


def results(request):
    file_name = request.GET.get('file')
    if not file_name:
        return render(request, 'selector/results.html', {'error': 'لا توجد نتائج للعرض بعد.'})
    results_path = settings.MEDIA_ROOT / 'results' / file_name
    if not os.path.exists(results_path):
        return render(request, 'selector/results.html', {'error': 'النتائج غير موجودة.'})
    with open(results_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return render(request, 'selector/results.html', {'data': data})
