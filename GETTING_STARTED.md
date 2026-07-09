# دليل استكشاف المشروع والبدء بالتدريب

## 1. الفكرة الأساسية (شو بنعمل هاد الكود)

المشكلة: عندك جملة عربية فيها كيانين (مثلاً شخص ومنظمة)، وبدك تعرف "شو العلاقة بينهم؟" (يشتغل بها، عضو فيها، مقرها فيها...).

الحل هون بدل ما يبني موديل تصنيف عادي بـ40 صنف، بيحوّل المشكلة لسؤال Yes/No أبسط:

> "هل الجملة X **بتدعم** الفرضية Y؟"

مثال:
- الجملة (premise): "حبيب بولس مدير مال عكا أرسل رسالة..."
- الفرضية (hypothesis) المولّدة من قالب علاقة `Personal.has_occupation`: "حبيب بولس يعمل كـ / مهنته مدير مال عكا"
- الموديل بيجاوب: **True** (نعم، العلاقة موجودة)

## 2. تدفق البيانات الكامل (Pipeline) — بالترتيب

```
data/raw/train.jsonl (سجلات RE خام: sentence, subject, object, relation)
        │
        ▼  scripts/prepare_data.py
src/data/templates.py  →  القوالب (40 علاقة) + RELATION_SCHEMA (domain/range)
        │
        ▼
src/data/nli_generator.py  →  يولّد أزواج (premise, hypothesis, Label)
        │
        ▼
data/nli/{train,val,test}.jsonl
        │
        ▼  scripts/train.py
src/data/loader.py  →  يحمّل الـ JSONL لـ DataFrame
src/data/dataset.py  →  يحوّلها لـ PyTorch Dataset (تحويل نص → tokens)
        │
        ▼
src/training/cross_val.py  →  حلقة K-Fold + fine-tuning لـ ARBERTv2
src/models/trainer.py  →  دالة الخسارة L_WCE + L_NCE
src/evaluation/metrics.py  →  Micro-F1 لاختيار أفضل نموذج بكل fold
        │
        ▼
outputs/cls_train_{fold}/best_model/  →  نموذج محفوظ لكل fold
        │
        ▼  scripts/predict.py
src/evaluation/metrics.py::run_ensemble_inference  →  متوسط تنبؤات كل الـ folds
```

## 3. شرح أهم جزئية ذكية بالمشروع: توليد الـ negatives (src/data/nli_generator.py)

هاي الجزئية الأهم لفهمها لأنها بتأثر مباشرة على جودة الداتا وبالتالي الدقة. بدل ما ياخد negatives عشوائية، فيه استراتيجيتين ذكيتين:

1. **Ambiguous hard negatives**: لكل triple إيجابي (مثلاً `manager_of`)، بياخد علاقة "شقيقة" مشابهة بالمعنى من `AMBIGUOUS_GROUPS` (مثلاً `leader_of` أو `president_of`) ويطبقها على نفس الكيانين كمثال سلبي. هاد بيخلي الموديل يتعلم يميز بدقة بين علاقات متشابهة، مش بس يميز عشوائي.
2. **قبل ما ياخد أي negative، بيتأكد إنه مو صح فعلياً** عن طريق 3 فحوصات أمان (`_is_safe_negative`): exact match، symmetry (علاقات متماثلة زي "أخ")، و transitive inference (علاقات احتوائية زي "يقع في").

هاي النقطة مهمة: **جودة الـ negatives = جودة تمييز الموديل**. إذا الـ negatives سهلة/عشوائية كتير، الموديل بتعلم شغلة سطحية.

## 4. كيف تبلش عملياً (خطوة خطوة)

```bash
pip install -r requirements.txt
```

### أ) جرّب أول بالعينة الجاهزة (بدون داتا كاملة)
```bash
python scripts/train.py --config configs/config.yaml
python scripts/predict.py --config configs/config.yaml
```
هيك بتشوف الـ pipeline كامل شغال خلال دقائق (280 مثال بس، فسريع)، وبتفهم شكل الـ output بـ `outputs/`.

### ب) لما تحصل على الداتا الكاملة (WojoodRelations)
```bash
# ضع الملف الخام في data/raw/train.jsonl
python scripts/prepare_data.py --config configs/config.yaml   # يولّد data/nli/*.jsonl
python scripts/train.py --config configs/config.yaml
python scripts/predict.py --config configs/config.yaml
```

## 5. كيف توصل لأعلى دقة ممكنة — دليل عملي مبني على الكود

هاي أهم نقطة، وهي مرتبطة مباشرة بـ configs/config.yaml:

### أ) حجم الداتا (الأهم على الإطلاق)
العينة المرفقة صغيرة جداً (280/60/60) — دقة أي موديل عليها مش تمثيلية. **أول شي لازم تسوّيه هو تحمّل الداتا الكاملة (WojoodRelations)** وتشغّل `prepare_data.py` عليها. زيادة `n_positive`/`n_negative` بـ config لما تصير الداتا الخام أكبر:
```yaml
data:
  n_positive: 2000   # بدل 200
  n_negative: 2000
```

### ب) موازنة الـ Loss (`loss:` في config.yaml)
- `class_weights: [w_n, w_p]` — إذا لسا الموديل بيفوّت حالات True كتير (recall منخفض للـ positive)، زيد `w_p` (مثلاً من 1.0 لـ 1.5-2.0). إذا صار بيتنبأ True كتير غلط، زيد `w_n`.
- `tau` (درجة حرارة الـ NCE): قيمة أقل (مثلاً 0.5) بتخلي الموديل "أكثر ثقة/حدة" بالفرق بين الأصناف؛ جرب قيم بين 0.5–2.0.

### ج) عدد الـ folds و epochs (`model.num_folds`, `training.num_train_epochs`)
حالياً `num_folds: 2` و `num_train_epochs: 2` — هاي قيم قليلة جداً (بتبدو معمولة للتجربة السريعة، مش للنتيجة النهائية). لأعلى دقة:
```yaml
model:
  num_folds: 5        # K-Fold أكبر = تقدير أقوى + ensemble أقوى بالـ predict
training:
  num_train_epochs: 5-10   # راقب overfitting عبر eval curve
```
تذكر: كل fold بيحفظ موديل، و`predict.py` بيعمل **ensemble** (متوسط) لكل الـ folds تلقائياً — فزيادة `num_folds` بتحسّن نتيجة الـ ensemble مباشرة.

### د) max_len
`max_len: 128` — شغّل `dataset.report_truncation()` (موجودة بـ src/data/dataset.py) على بياناتك لتتأكد ما في جمل مقطوعة (truncated) بتخسّرك معلومات، خصوصاً إذا الجمل الأصلية طويلة.

### هـ) learning_rate و batch size
`learning_rate: 2e-5` قيمة معقولة قياسية لـ fine-tuning BERT-family، بس جرب `1e-5` و `3e-5` وقارن.

### و) جودة القوالب والـ schema (templates.py)
إذا لاحظت الموديل بيغلط بعلاقات معينة بشكل متكرر، ارجع لـ `AMBIGUOUS_GROUPS` بـ src/data/templates.py وتأكد إنها شاملة كل الازواج اللي فعلياً بتلخبط الموديل — ضيف مجموعات جديدة إذا لقيت أنماط غلط متكررة بـ `false_predictions.jsonl` (الملف اللي بيطلعه `predict.py`).

### ز) حلقة تحسين عملية
1. درّب → شغّل `predict.py`
2. افتح `outputs/false_predictions.jsonl` وحلل أخطاء الموديل يدوياً
3. صنّف الأخطاء: هل هي بسبب قوالب متشابهة؟ negatives غير كافية؟ بيانات قليلة لعلاقة معينة؟
4. عدّل القوالب/الـ config حسب النتيجة، وكرر
