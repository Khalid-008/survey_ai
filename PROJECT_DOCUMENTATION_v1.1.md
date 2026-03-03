# 📊 Survey AI — توثيق المشروع الشامل
**الإصدار:** v1.1  
**التاريخ:** 2026-03-03  
**المشروع:** منصة تحليل الاستبيانات بالذكاء الاصطناعي

---

## 1. نظرة عامة على المشروع

**Survey AI** هو نظام ذكاء اصطناعي متكامل لتحليل الاستبيانات تلقائياً. يستقبل المشروع بيانات استبيان معيّن من قاعدة بيانات MySQL، ثم يُشغّل سلسلة من العمليات المتقدمة لاستخراج رؤى تنفيذية ورسوم بيانية تفاعلية، وعرضها على لوحة تحكم بصرية احترافية.

### الهدف الأساسي
تحويل إجابات الاستبيانات الخام إلى **تقرير تنفيذي بالعربية** + **5 رسوم بيانية تفاعلية** بشكل تلقائي دون تدخل بشري.

---

## 2. هيكل المجلدات

```
survey_ai/
├── src/                          # الكود الخلفي (Backend - Python)
│   ├── app.py                    # نقطة دخول Flask API
│   ├── agents/                   # وكلاء اللانغ‌غراف
│   │   ├── graphs/
│   │   │   ├── nodes.py          # تعريف الـ 5 نودز الرئيسية
│   │   │   ├── workflow.py       # بناء الـ LangGraph وتشغيله
│   │   │   └── setup.py          # تعريف الـ State والذاكرة
│   │   └── prompt/               # قوالب البرومبتات
│   │       ├── synthesis_agent_prompt.py
│   │       ├── chart_generation_prompt.py
│   │       └── selection_analytics_prompt.py
│   ├── analytics_pipeline/       # خطوط معالجة البيانات
│   │   ├── selection_pipeline.py # تحليل أسئلة الخيارات عبر SQL
│   │   ├── text_pipeline.py      # تحليل الأسئلة النصية (NLP)
│   │   ├── data_processing/
│   │   │   ├── data_cleaner.py   # تنظيف البيانات المكررة
│   │   │   └── data_enricher.py  # تطبيق NLP على الأسئلة النصية
│   │   └── util/                 # أدوات مساعدة
│   │       ├── selection_utils.py
│   │       ├── synthesis_utils.py
│   │       └── chart_utils.py
│   ├── data/
│   │   └── operations.py         # تعامل مع قاعدة بيانات MySQL
│   ├── llms/
│   │   └── models.py             # تعريف نموذج اللغة (LLM)
│   ├── tests/
│   │   └── chart_agent_snapshot.json  # snapshot للاختبار
│   └── exports/                  # التقارير المُصدَّرة
├── vue-project/                  # الواجهة الأمامية (Frontend)
│   ├── csat_dashboard_primevue.html  # لوحة التحكم الرئيسية
│   ├── test_charts.html          # صفحة اختبار الرسوم
│   └── src/                      # مكونات Vue.js
├── exports/                      # ملفات التقارير المحفوظة
└── logs/                         # سجلات التشغيل
```

---

## 3. المكونات الرئيسية

### 3.1 Flask API — `src/app.py`

نقطة الدخول الرئيسية للنظام. يُوفّر 3 نقاط نهاية (endpoints):

| المسار | الطريقة | الوصف |
|--------|---------|-------|
| `/survey_insight` | POST | المسار الرئيسي — يشغّل سير العمل الكامل |
| `/test_charts` | GET | اختبار توليد الرسوم فقط من الـ snapshot |
| `/test_full` | GET | اختبار التوليف + الرسوم من الـ snapshot |

**نموذج الطلب لـ `/survey_insight`:**
```json
{
  "request": {
    "survey_id": "S25120024",
    "message": "حلل نتائج الاستبيان",
    "session_id": "abc123",
    "date_from": "2025-01-01",   // اختياري
    "date_to": "2025-12-31"      // اختياري
  }
}
```

---

### 3.2 LangGraph Workflow — `src/agents/graphs/workflow.py`

النظام مبني على **LangGraph** لتنظيم تسلسل العمليات. يسير الـ workflow بترتيب خطي ثابت عبر **5 نودز**:

```
START
  ↓
[1] retrieve_survey_question
  ↓
[2] analyze_selection_questions
  ↓
[3] analyze_text_questions
  ↓
[4] synthesis_agent
  ↓
[5] generate_charts_agent
  ↓
END
```

الـ workflow يدعم **حفظ الجلسات** (`checkpointer=memory`) حيث يُميّز كل جلسة بـ `session_id`.

---

### 3.3 النودز الخمسة — `src/agents/graphs/nodes.py`

#### 🟦 NODE 1: `retrieve_survey_question`
**المهمة:** جلب بيانات الاستبيان الخام من MySQL.

- يستقبل `survey_id` + فلتر تاريخي اختياري (`date_from`, `date_to`)
- يُنفّذ استعلام SQL لجلب إجابات المستخدمين
- يُخزّن البيانات كـ DataFrame في `state["survey_data"]`

---

#### 🟩 NODE 2: `analyze_selection_questions`
**المهمة:** تحليل أسئلة الخيارات (غير النصية) بتوليد SQL ذكي.

يستدعي **Selection Pipeline** الذي يعمل كالتالي:
1. جلب بيانات أسئلة الخيارات من MySQL
2. تجميع الإجابات حسب كل سؤال
3. إرسال معلومات الأسئلة إلى نموذج اللغة (LLM)
4. النموذج يُولّد **5 كويريات SQL تحليلية**
5. تنفيذ الكويريات وتخزين النتائج

أنواع الأسئلة المدعومة: `MULTIPLE_CHOICE`, `DROPDOWN`, `NPS`, `EMOJIS`, `YES_NO`

---

#### 🟨 NODE 3: `analyze_text_questions`
**المهمة:** تحليل الأسئلة النصية المفتوحة (`TEXT_INPUT`) بتقنيات NLP متقدمة.

**خطوات الإثراء:**

| الخطوة | الأداة | النموذج المستخدم |
|--------|-------|-----------------|
| 🧹 تنظيف | `data_cleaner.py` | — |
| 💬 تحليل المشاعر | Hugging Face Pipeline | `CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment` |
| 🏷️ استخراج الكيانات (NER) | GLiNER | `NAMAA-Space/gliner_arabic-v2.1` |
| 🔍 استخراج المواضيع | BERTopic + UMAP + HDBSCAN | `CAMeL-Lab/bert-base-arabic-camelbert-da` |

**الكيانات المستخرجة:** شخص، مكان، منظمة، تاريخ، وقت، منتج، حدث

**ملاحظة مهمة:** إذا لم يوجد `TEXT_INPUT` في الاستبيان تُتخطى هذه المرحلة تلقائياً.

---

#### 🟥 NODE 4: `synthesis_agent`
**المهمة:** توليد التقرير التنفيذي الشامل باللغة العربية.

- يجمع نتائج النودز 2 و3
- يُرسلها إلى نموذج اللغة مع برومبت متخصص
- يُرجع **JSON منظّم** يحتوي على:

```json
{
  "executive_summary": "ملخص تنفيذي 2-3 جمل",
  "detailed_analysis": "تحليل تفصيلي بالعربية",
  "key_metrics": ["مقياس 1", "مقياس 2", "..."],
  "recommendations": ["توصية 1", "توصية 2", "..."],
  "visualizations": []
}
```

- يحفظ التقرير في `src/exports/` و`exports/`
- يحفظ **snapshot** في `tests/chart_agent_snapshot.json` للاختبار

---

#### 🟪 NODE 5: `generate_charts_agent`
**المهمة:** توليد إعدادات 5 رسوم بيانية مناسبة باستخدام Chart.js.

- يستقبل ملخص التحليل من النودز السابقة
- يُرسله للنموذج اللغوي مع برومبت متخصص
- يُولّد **5 إعدادات** بتنسيق `Chart.js` جاهزة للواجهة الأمامية
- إذا جاء أكثر من 5 رسوم يقطع الزيادة تلقائياً

---

### 3.4 برومبتات النظام — `src/agents/prompt/`

#### `synthesis_agent_prompt.py`
- **النوع:** System Prompt + Human Prompt مُدمجان
- **اللغة:** الإخراج بالعربية حصراً
- **القيود:** لا أسماء أشخاص، لا أكواد تقنية، لا مصطلح NPS بالإنجليزية
- **الإخراج:** JSON صارم بدون markdown

#### `selection_analytics_prompt.py`
- **المهمة:** توليد SQL لتحليل أسئلة الخيارات
- **القيود:** قواعد SQL صارمة (GROUP BY، CAST، window functions)
- **الإخراج:** 5 كويريات SQL تحليلية

#### `chart_generation_prompt.py`
- **المهمة:** توليد إعدادات Chart.js
- **الإخراج:** JSON لـ 5 رسوم بيانية (bar, pie, line, doughnut, etc.)

---

### 3.5 معالجة النصوص العربية — `src/analytics_pipeline/text_pipeline.py`

ملف ضخم (962 سطر) يضم:

- **`normalize_arabic()`** — تطبيع النص العربي (الأحرف، المسافات، الرموز)
- **`convert_arabic_time_to_24h()`** — تحويل الوقت العربي (م/ص) إلى 24 ساعة
- **`remove_stopwords()`** — إزالة الكلمات الشائعة غير المفيدة
- **`analyze_sentiment_batch()`** — تحليل مشاعر على دفعات
- **`extract_entities_batch()`** — استخراج كيانات مُسمّاة
- **`extract_topics_from_texts()`** — استخراج مواضيع بـ BERTopic
- **`enrich_survey_data()`** — الدالة الرئيسية التي تُطبّق الـ 3 تحليلات تسلسلياً

**تحسين الأداء:** النماذج تُحمَّل مرة واحدة وتُخزّن في الذاكرة (`_cached_*` globals).

---

### 3.6 معالجة ودة الاستبيان — `src/data/operations.py`

يُوفّر دوال للتواصل مع MySQL:
- `get_survey_df()` — جلب كل بيانات الاستبيان
- `get_selection_questions_data()` — جلب أسئلة الخيارات فقط
- `execute_raw_query()` — تنفيذ أي كويري SQL مُولَّد

جميع الدوال تدعم **فلتر التاريخ** (`date_from`, `date_to`).

---

### 3.7 الواجهة الأمامية — `vue-project/`

**التقنيات:** Vue.js 3 (CDN) + PrimeVue + Chart.js

**الملفات الرئيسية:**

| الملف | الوصف |
|-------|-------|
| `csat_dashboard_primevue.html` | لوحة التحكم الرئيسية الإنتاجية |
| `test_charts.html` | صفحة اختبار الرسوم البيانية |
| `src/` | مكونات Vue.js المنفصلة |

**ميزات لوحة التحكم:**
- 📄 **الملخص التنفيذي** — عرض موجز
- 📊 **التحليل التفصيلي** — accordion قابل للطي
- 📈 **المقاييس الرئيسية** — بطاقات إحصائية
- 💡 **التوصيات** — قائمة قابلة للطي
- 📉 **الرسوم البيانية** — 5 رسوم تفاعلية بـ Chart.js
- 📅 **فلتر التاريخ** — تصفية حسب نطاق زمني

---

## 4. تدفق البيانات الكامل

```
المستخدم يرسل طلب POST /survey_insight
         │
         ▼
    Flask app.py
         │
         ▼
  LangGraph Workflow
         │
    ┌────┴────┐
    │  NODE 1  │  جلب بيانات MySQL
    └────┬────┘
         │
    ┌────┴────┐
    │  NODE 2  │  Selection Pipeline
    │         │  → LLM يُولّد SQL
    │         │  → تنفيذ 5 كويريات
    └────┬────┘
         │
    ┌────┴────┐
    │  NODE 3  │  Text NLP Pipeline
    │         │  → Sentiment Analysis
    │         │  → NER
    │         │  → Topic Modeling
    └────┬────┘
         │
    ┌────┴────┐
    │  NODE 4  │  Synthesis Agent
    │         │  → LLM يُولّد تقرير JSON
    │         │  → حفظ snapshot
    └────┬────┘
         │
    ┌────┴────┐
    │  NODE 5  │  Chart Generation Agent
    │         │  → LLM يُولّد 5 charts
    └────┬────┘
         │
         ▼
   JSON Response
   {
     executive_summary, detailed_analysis,
     key_metrics, recommendations,
     charts: [5 chart configs]
   }
         │
         ▼
   Vue.js Dashboard
   (عرض بصري للمستخدم)
```

---

## 5. نماذج الذكاء الاصطناعي المستخدمة

| النموذج | الغرض | المصدر |
|---------|-------|--------|
| LLM (مُعرَّف في `llms/models.py`) | توليد SQL، تلخيص، رسوم | LangChain |
| `bert-base-arabic-camelbert-msa-sentiment` | تحليل المشاعر العربية | CAMeL-Lab / HuggingFace |
| `gliner_arabic-v2.1` | استخراج الكيانات (NER) | NAMAA-Space |
| `bert-base-arabic-camelbert-da` | Embeddings للمواضيع | CAMeL-Lab / HuggingFace |
| BERTopic + UMAP + HDBSCAN | تجميع المواضيع | مكتبات Python |

---

## 6. الإخراج والتقارير

### 6.1 JSON Response (الاستجابة الفورية)
يُعاد من `/survey_insight` مباشرة ويُعرض على الواجهة.

### 6.2 ملفات مُصدَّرة (Exports)
- **`src/exports/`** — تقارير التلخيص النصية `.txt`
- **`exports/`** — في المجلد الجذر للمشروع
- **`src/tests/chart_agent_snapshot.json`** — snapshot لاختبار نود الرسوم

### 6.3 سجلات الـ Selection Pipeline
يُحفظ في مجلد `logs/` كنتيجة اختبار لكل كويري SQL مُنفَّذ.

---

## 7. كيفية التشغيل

### تشغيل الخادم الخلفي (Backend)
```bash
cd c:\Users\Khalid PC\Desktop\Channels\survey_ai\src
python app.py
# يعمل على: http://0.0.0.0:5566
```

### تشغيل الواجهة الأمامية (Frontend)
```bash
cd c:\Users\Khalid PC\Desktop\Channels\survey_ai\vue-project
npm run dev
# يعمل على: http://localhost:5173
```

---

## 8. ملاحظات تقنية مهمة

### 8.1 معالجة الأخطاء
- كل نود يُطبّق **graceful error handling**: في حالة الفشل يُكمل الـ workflow بالبيانات المتاحة
- NODE 2 و NODE 3: فشلهما لا يُوقف الـ workflow
- NODE 1: فشله يُوقف كل شيء (لا بيانات → لا تحليل)

### 8.2 فلتر التاريخ
- اختياري في كل مكان
- يُمرَّر عبر سلسلة كاملة: `app.py → workflow → nodes → data/operations.py`

### 8.3 الجلسات والذاكرة
- `session_id` يُحدد thread مستقل في LangGraph checkpointer
- يُتيح إعادة استخدام بيانات جلسة سابقة

### 8.4 تحسين أداء النماذج
- نماذج Hugging Face تُحمَّل مرة واحدة عند أول طلب وتبقى في الذاكرة
- `CHUNK_SIZE = 5000` لمعالجة الدفعات الكبيرة دون مشاكل ذاكرة

---

## 9. تاريخ التطوير المختصر

| الإصدار | التغييرات الرئيسية |
|---------|-------------------|
| v0.1 | بنية أساسية: Flask + LangGraph + أسئلة نصية |
| v0.5 | إضافة Selection Pipeline + SQL generation |
| v0.8 | Synthesis Agent + Chart Generation Agent |
| v0.9 | Vue.js Dashboard + PrimeVue |
| v1.0 | Date Filter + إصلاح رسم الـ charts |
| **v1.1** | **هذا التوثيق الشامل** |

---

*آخر تحديث: 2026-03-03 | كتب بواسطة: Antigravity AI*
