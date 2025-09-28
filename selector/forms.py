from django import forms


class UploadDatasetForm(forms.Form):
    file = forms.FileField(label='ملف CSV')
    target_column = forms.CharField(max_length=100, label='اسم عمود الهدف')
    task = forms.ChoiceField(
        label='نوع المسألة',
        choices=[('auto', 'تلقائي'), ('classification', 'تصنيف'), ('regression', 'انحدار')],
        initial='auto'
    )

    population_size = forms.IntegerField(min_value=5, max_value=2000, initial=30, label='حجم المجتمع')
    generations = forms.IntegerField(min_value=5, max_value=1000, initial=20, label='عدد الأجيال')
    crossover_rate = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.8, label='معدل العبور')
    mutation_rate = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.05, label='معدل الطفرة')
    elitism = forms.IntegerField(min_value=0, max_value=100, initial=2, label='عدد الأفراد بالنخبة')
    max_features = forms.IntegerField(required=False, min_value=1, label='أقصى عدد للميزات (اختياري)')
    scoring = forms.ChoiceField(
        label='مقياس التقييم',
        choices=[
            ('auto', 'تلقائي'),
            ('accuracy', 'Accuracy'),
            ('f1', 'F1'),
            ('roc_auc', 'ROC AUC'),
            ('r2', 'R2'),
            ('neg_mean_squared_error', '-MSE'),
        ],
        initial='auto'
    )
    cv = forms.IntegerField(min_value=2, max_value=20, initial=5, label='عدد الطيات (CV)')
