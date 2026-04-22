# Teaching notes for the professor

## Suggested 25-minute explanation flow

### Part 1 — Motivation (5 minutes)
- ابدأ من مشكلة التصيد المتقدم: الرسائل اليوم ليست ركيكة كما كانت سابقًا.
- وضّح أن المهاجم لم يعد يعتمد فقط على الرابط، بل على اللغة المقنعة أيضًا.
- اربط ذلك مباشرة بفكرة sentiment + persuasion.

### Part 2 — Architecture (7 minutes)
- اشرح المسار النصي: تمثيل الرسالة عبر Transformer أو LLM.
- اشرح المسار الشعوري: urgency, threat, reward, action, URLs.
- اشرح الدمج النهائي قبل طبقة التصنيف.

### Part 3 — Code walkthrough (6 minutes)
- `src/train_baseline.py`
- `src/train_hybrid.py`
- `src/train_transformer.py`
- `src/llm_prompt_eval.py`

### Part 4 — Results & limitations (4 minutes)
- أظهر أن النتائج الحالية على البيانات الصناعية ليست benchmark نهائيًا.
- شدد على أهمية البيانات الحقيقية وتحليل الأخطاء.
- وضّح قيمة النموذج الهجين في التفسير، حتى لو لم يتحسن F1 كثيرًا على البيانات السهلة.

### Part 5 — Discussion questions (3 minutes)
- هل الدقة وحدها تكفي؟
- لماذا يعد Recall مهمًا في الأمن السيبراني؟
- ما أثر تسرب البيانات بين التدريب والاختبار؟
- هل نستخدم LLM مباشرة للتصنيف أم كمولّد Features؟

## Classroom demo commands

```bash
pip install -r requirements.txt
bash scripts/run_demo.sh
```

## Master-level extensions
- Fine-tune RoBERTa on a real phishing dataset.
- Compare with prompt-based local LLM.
- Add explainability via SHAP or coefficient inspection.
- Build a multilingual extension for Arabic + English emails.
