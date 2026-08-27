# توثيق مشروع: كشف اختراق الشبكات (Network Intrusion Detection) - NSL-KDD

## 📌 نظرة عامة

**النوع:** Multi-class Classification
**الداتاسيت:** NSL-KDD (نسخة محسّنة من KDD Cup 99)
**حجم Train:** 125,973 صف × 42 عمود
**حجم Test:** 22,544 صف × 42 عمود (ملف منفصل من الأصل، مصمم بتحديات إضافية)
**الهدف:** تصنيف اتصال الشبكة إلى: `normal`, `DoS`, `Probe`, `R2L`, `U2R`

### رابط الداتا
- نسخة CSV جاهزة (بدون تسجيل): `https://raw.githubusercontent.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/master/NSL_KDD_Train.csv` و`NSL_KDD_Test.csv`
- المصدر الرسمي (يتطلب تسجيل): `https://www.unb.ca/cic/datasets/nsl.html`

---

## 🗂️ مراحل المشروع

### 1) تحميل البيانات وإضافة أسماء الأعمدة
الملف الخام بدون Header، فأضفنا أسماء الـ 41 عمود + `label` يدويًا (أسماء معيارية موثقة من مصممي الداتاسيت).

### 2) تجميع أنواع الهجمات (23 نوع → 5 فئات)
استخدمنا `dict` + `.map()` لتحويل الأنواع التفصيلية (neptune, smurf, satan...) إلى 5 فئات كبرى معتمدة بالأبحاث الأكاديمية:
- **DoS**: neptune, smurf, back, teardrop, pod, land
- **Probe**: satan, ipsweep, portsweep, nmap
- **R2L**: warezclient, guess_passwd, warezmaster, imap, ftp_write, multihop, phf, spy
- **U2R**: buffer_overflow, rootkit, loadmodule, perl

**سبب التجميع:** التصنيف لـ23 فئة مستحيل عمليًا لأن بعض الفئات (spy: 2 صف، perl: 3 صف) لا تحتوي أمثلة كافية للتعلم منها.

### 3) اكتشاف مشكلة "Unknown" بملف Test
3,750 صف (16.6%) من Test رجعوا NaN بعد التجميع — أنواع هجمات غير موجودة إطلاقًا بـ Train. عولجت بـ `fillna('Unknown')` كفئة سادسة منفصلة، بهدف **مراقبة** سلوك الموديل أمام هجمات غير مسبوقة (Zero-day)، وليس تصنيفها بدقة.

### 4) بناء Pipeline لمنع Data Leakage
**القاعدة الذهبية المتبعة:** تقسيم البيانات (أو استخدام ملفات Train/Test منفصلة من الأصل) **قبل** أي معالجة تتعلم من توزيع البيانات (Encoding, Scaling, SMOTE).

```python
preprocessor = ColumnTransformer([
    ("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")
```

- `OneHotEncoder` بدل `Ordinal Encoding` لأن الأعمدة الفئوية (protocol_type, service, flag) **غير مرتبة منطقيًا** (بعكس مشروع الألماس).
- `handle_unknown="ignore"` يمنع كسر الكود لو ظهرت قيمة بـ Test غير موجودة بـ Train.
- الـ `.fit()` يتعلّم فقط من `X_train`؛ `.predict()` يطبّق نفس التحويل على `X_test` دون أي تسريب معلومات.

### 5) Baseline: Random Forest بدون معالجة Imbalance

| المقياس | القيمة |
|---|---|
| Accuracy | 0.7229 |
| R2L recall | 0.06 |
| U2R recall | 0.05 |

**المشكلة المكتشفة:** Accuracy عام يبدو مقبولًا، لكنه مضلل — الموديل يتجاهل الفئات النادرة (R2L, U2R) شبه كليًا لأن تجاهلها "يقلل الخطأ الكلي" إحصائيًا.

### 6) معالجة Imbalanced Data — أربع محاولات

| الطريقة | R2L recall | U2R recall | الخلاصة |
|---|---|---|---|
| Baseline (بدون معالجة) | 0.06 | 0.05 | نقطة البداية |
| `class_weight='balanced'` | 0.01 | 0.03 | **ساء** — التوزين لا يعوّض غياب التنوع داخل Bootstrap |
| `class_weight='balanced_subsample'` | أسوأ من balanced | أسوأ | نفس المشكلة الجذرية: قلة العينات، لا علاقة بطريقة حساب الوزن |
| SMOTE (بعد OneHotEncoding) | 0.01 | **0.14** | يحسّن U2R لكن يضر R2L — الاستيفاء الرقمي على أعمدة One-Hot ينتج قيمًا كسرية بلا معنى (مثل 0.5 لعمود ثنائي) |
| **SMOTENC** (قبل Encoding، يفهم الأعمدة الفئوية) | **0.12** | 0.08 | أفضل توازن عام، Accuracy = 0.7323 |

### 7) اكتشاف السبب الجذري لضعف R2L (Distribution Shift)

بالتحقق من الأنواع الفرعية:
- **Train**: R2L مكوّنة بنسبة ~89% من نوع واحد (`warezclient`: 890 من 995 صف).
- **Test**: `warezclient` **غير موجود إطلاقًا**؛ كل صفوف R2L من الأنواع النادرة جدًا بـ Train (guess_passwd, warezmaster, imap...).

**الاستنتاج:** المشكلة ليست فقط "قلة عدد" (Imbalance)، بل **تغيّر جذري بتوزيع الأنماط الفرعية** بين Train وTest (Distribution Shift) — وهذا تحدٍ متعمّد بتصميم NSL-KDD لمحاكاة الواقع (تطوّر أنماط الهجمات بمرور الوقت). أي موديل يتعلم من نفس البيانات المحدودة سيتعثر بنفس النقطة تقريبًا، بغض النظر عن قوته.

### 8) مقارنة الموديلات

| الموديل | Accuracy | R2L recall | U2R recall | ملاحظات |
|---|---|---|---|---|
| Random Forest + SMOTENC | 0.7323 | 0.12 | 0.08 | أفضل توازن عام |
| Decision Tree + SMOTENC | 0.7125 | 0.09 | **0.32** | أضعف Accuracy عام، لكن أفضل recall لـU2R تحديدًا |

**SVM لم يُجرَّب** — بسبب التعقيد الزمني العالي (~O(n²) إلى O(n³)) غير العملي على 125,973 صف بدون تصغير العينة أو استخدام `LinearSVC`.

---

## 🎓 أهم الدروس المستفادة

1. **Accuracy وحدها مضللة تمامًا مع بيانات غير متوازنة** — لازم `classification_report` كامل (precision/recall/f1 لكل فئة) + `macro avg` (يعكس أداء الفئات الصغيرة بعدل، بعكس `weighted avg`).

2. **Recall أهم من Precision بمجال أمن الشبكات** — فوات هجوم حقيقي (False Negative) أخطر بكثير من إنذار كاذب (False Positive).

3. **Data Leakage له أشكال متعددة ودقيقة:**
   - دمج Train/Test لعمل Encoding سوا = تسريب خفيف (حُلّ بـ `OneHotEncoder(handle_unknown='ignore')` مع `.fit()` على Train فقط).
   - تطبيق SMOTE قبل التقسيم = تسريب خطير جدًا (صفوف اصطناعية مبنية من بيانات قد "تسرّب" لـ Test).

4. **`class_weight` له حدود حقيقية** — يعيد توزين الأخطاء رياضيًا، لكن لا يولّد بيانات جديدة. مع Bootstrap Sampling، الفئات النادرة جدًا قد لا تظهر أصلًا بعينة كل شجرة، فالتوزين لا يعالج المشكلة الجذرية.

5. **SMOTE العادي لا يصلح للبيانات المختلطة (رقمية + فئوية)** — يُنتج قيمًا كسرية بلا معنى على أعمدة One-Hot. الحل: `SMOTENC` مع ترتيب Pipeline بحيث يُطبَّق **قبل** الـ Encoding.

6. **معالجة Imbalance لها سقف** إذا كانت المشكلة الحقيقية Distribution Shift (تغيّر الأنماط الفرعية بين Train/Test) وليست فقط قلة عدد — عندها تغيير الموديل أو تقنية الموازنة لن يحل المشكلة الجذرية.

7. **اختيار الموديل له Trade-offs حسب الأولوية:** Random Forest أعطى أفضل توازن عام، لكن Decision Tree (رغم Accuracy أضعف) أعطى أفضل Recall لـU2R تحديدًا — يعني "الأفضل" يعتمد على أي فئة أهم عمليًا.

---

## 📊 جدول ملخص كل التجارب

| # | التجربة | Accuracy | R2L recall | U2R recall | Unknown recall |
|---|---|---|---|---|---|
| 1 | Random Forest (يدوي، بدون Pipeline) | 0.7188 | 0.03 | 0.00 | 0.00 |
| 2 | Random Forest + Pipeline (OneHotEncoder نظيف) | 0.7229 | 0.06 | 0.05 | 0.00 |
| 3 | + `class_weight='balanced'` | 0.7213 | 0.01 | 0.03 | 0.00 |
| 4 | + `class_weight='balanced_subsample'` | أسوأ | أسوأ | أسوأ | 0.00 |
| 5 | + SMOTE (بعد Encoding) | 0.7224 | 0.01 | 0.14 | 0.00 |
| 6 | + **SMOTENC** (قبل Encoding) | **0.7323** | **0.12** | 0.08 | 0.00 |
| 7 | Decision Tree + SMOTENC | 0.7125 | 0.09 | **0.32** | 0.00 |

*ملاحظة: Unknown recall = 0.00 بكل التجارب، وهذا متوقع تمامًا — الموديل لم يشاهد هذه الفئة إطلاقًا أثناء التدريب بتصميم الداتاسيت.*

---

## 🔧 الأدوات والمكتبات المستخدمة
`pandas`, `scikit-learn` (Pipeline, ColumnTransformer, OneHotEncoder, RandomForestClassifier, DecisionTreeClassifier, classification_report), `imbalanced-learn` (SMOTE, SMOTENC, imblearn.Pipeline)
