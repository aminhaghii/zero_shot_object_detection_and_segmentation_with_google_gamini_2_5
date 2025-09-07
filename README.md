# شناسایی و بخش‌بندی شیء به‌صورت Zero-Shot با Google Gemini 2.5

![Pipeline](assets/pipeline.svg)

این مخزن یک راهنمای عملی و نظام‌مند برای پیاده‌سازی شناسایی (Object Detection) و بخش‌بندی (Segmentation) بدون‌نیاز به داده‌های برچسب‌خورده (Zero-Shot) با استفاده از مدل‌های بینایی Gemini است. ایده‌ی اصلی این است که به‌جای آموزش یک مدل اختصاصی، از قدرت استنتاج مدل‌های چندوجهی (Vision-Language) استفاده کنیم و با تعریف «پرامپت ساخت‌یافته» از مدل بخواهیم جعبه‌های مرزی (Bounding Boxes) و نقاب‌ها/ماسک‌ها (Polygons/Masks) را برگرداند.

توجه: اگرچه در این مخزن از اصطلاح «Gemini 2.5» استفاده می‌شود، کد نمونه بر اساس SDK فعلی مطرح شده است. بسته به دسترسی شما، می‌توانید نام مدل را به نسخه دقیقِ در دسترس (مثل gemini-1.5-pro/flash یا نسخه‌های جدیدتر) تغییر دهید.

---

## ویژگی‌ها
- Zero-Shot: بدون نیاز به دیتاست برچسب‌خورده یا Fine-Tune
- یکپارچگی تشخیص و بخش‌بندی: خروجی شامل bbox و polygon برای هر کلاس
- قابل توسعه برای چندین کلاس با یک پرامپت ساخت‌یافته
- قابل اتصال به هر منبع تصویر (فایل، URL، فریم‌های ویدئو)
- رعایت امنیت: استفاده از متغیر محیطی برای کلید API

---

## نحوه کار (Concept)
1. ورودی تصویر (Image) + فهرست کلاس‌های مدنظر به‌صورت متن
2. ساخت پرامپت ساخت‌یافته: تعریف دقیق قالب خروجی (JSON با bbox و polygon)
3. ارسال تصویر و پرامپت به مدل بینایی Gemini
4. دریافت پاسخ ساخت‌یافته (Structured) و اعتبارسنجی JSON
5. پس‌پردازش: ترسیم نتایج روی تصویر، ذخیره به‌صورت فایل JSON/تصویر

شِمای خروجی پیشنهادی (JSON):
```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": [x, y, width, height],
      "polygon": [[x1, y1], [x2, y2], ...]
    }
  ],
  "image_size": {"width": 1280, "height": 720}
}
```

---

## پیش‌نیازها
- Python 3.9+
- دسترسی به Google Generative AI (کلید API)
- کتابخانه‌های پایه برای پردازش تصویر (opencv-python, pillow)
- کتابخانه Google Generative AI (google-generativeai)

نکته امنیتی: هرگز کلید API را در مخزن قرار ندهید. از متغیر محیطی استفاده کنید:
- Windows (PowerShell):
  ```powershell
  setx GEMINI_API_KEY "YOUR_KEY_HERE"
  ```
- Linux/Mac (bash):
  ```bash
  export GEMINI_API_KEY="YOUR_KEY_HERE"
  ```

---

## نصب
شما می‌توانید بسته‌های موردنیاز را به‌صورت دستی نصب کنید:
```bash
pip install google-generativeai pillow opencv-python numpy
```

---

## نمونه استفاده (Pseudo-code)
مثال زیر منطق کلی را نشان می‌دهد. نام مدل را مطابق دسترسی‌تان تنظیم کنید (مثلاً "gemini-1.5-pro" یا نسخه‌های جدیدتر):

```python
import os, json, base64
import google.generativeai as genai
from PIL import Image

API_KEY = os.getenv("GEMINI_API_KEY")
assert API_KEY, "GEMINI_API_KEY is not set"

genai.configure(api_key=API_KEY)
# اگر به Gemini 2.5 دسترسی دارید، نام مدل را به نسخه مربوطه تغییر دهید
MODEL_NAME = "gemini-1.5-pro"  # یا نسخه‌های جدیدتر در صورت دسترسی
model = genai.GenerativeModel(MODEL_NAME)

# 1) بارگذاری تصویر
img_path = "path/to/your/image.jpg"
image = Image.open(img_path).convert("RGB")

# 2) تعریف کلاس‌ها و فرمت خروجی
classes = ["person", "car", "dog"]
output_schema = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string"},
                    "confidence": {"type": "number"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "polygon": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    }
                },
                "required": ["class", "confidence", "bbox"]
            }
        },
        "image_size": {
            "type": "object",
            "properties": {
                "width": {"type": "number"},
                "height": {"type": "number"}
            },
            "required": ["width", "height"]
        }
    },
    "required": ["detections", "image_size"]
}

prompt = f"""
You are a zero-shot object detector and segmenter. Detect only these classes: {classes}.
Return a VALID JSON strictly matching the provided schema. Coordinates are in absolute pixel values.
For each detection provide: class, confidence (0..1), bbox=[x,y,width,height], and polygon as list of [x,y] points if available.
"""

# 3) ارسال تصویر و پرامپت (استفاده از multimodal input)
response = model.generate_content([
    prompt,
    image,
], generation_config={
    "response_mime_type": "application/json",
    "response_schema": output_schema
})

# 4) بازیابی و اعتبارسنجی پاسخ
result = json.loads(response.text)
print(json.dumps(result, ensure_ascii=False, indent=2))

# 5) پس‌پردازش و مصورسازی (نمونه ساده)
# می‌توانید با OpenCV bbox و polygon را روی تصویر رسم کنید و خروجی را ذخیره نمایید.
```

نکات مهم:
- برای کاهش خطا در ساختار JSON، حتما response_mime_type و response_schema را مشخص کنید.
- در صورت عدم دسترسی به نسخه‌های جدیدتر، از مدل‌های Vision موجود (مانند gemini-1.5-pro/flash) استفاده کنید و در آینده نام مدل را جایگزین نمایید.
- برای تصاویر بزرگ، بهتر است پیش‌پردازش (Resize/Pad) انجام دهید تا محدودیت‌های اندازه رعایت شود.

---

## ساختار مخزن
```
.
├── README.md
└── assets/
    └── pipeline.svg
```

شما می‌توانید اسکریپت‌ها، نوت‌بوک‌ها و ابزارهای جانبی خود را در این مخزن اضافه کنید (به‌عنوان مثال: notebooks/, scripts/, examples/). این README از ابتدا با هدف مستندسازی کامل و آماده‌سازی برای توسعه‌های بعدی فراهم شده است.

---

## بهترین‌روش‌ها (Best Practices)
- محرمانگی: کلیدها را هرگز در مخزن قرار ندهید؛ از .env یا متغیر محیطی استفاده کنید.
- پیاده‌سازی مرحله‌ای: ابتدا خروجی JSON معتبر بگیرید، سپس سراغ رسم و ارزیابی بروید.
- اعتبارسنجی: قبل از مصرف نتایج در زنجیره‌های بعدی، ساختار و محدوده مقادیر را بررسی کنید.
- ثبت نتایج: خروجی‌ها (تصویر+JSON) را برای ارزیابی‌های آتی آرشیو کنید.

---

## مجوز (License)
محتوای این مخزن آزاد است؛ اما هنگام استفاده از APIها، شرایط و قوانین سرویس‌دهنده (Google Generative AI) را رعایت کنید.

---

## مشارکت
پیشنهادها و بهبودها خوش‌آمدند. لطفاً Pull Request باز کنید یا Issue ثبت نمایید.