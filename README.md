# Zero‑Shot Object Detection & Segmentation with Google Gemini

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aminhaghii/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5/blob/main/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb)
[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/aminhaghii/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5/main/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb)

A practical, end‑to‑end guide to perform zero‑shot object detection and segmentation using Gemini Vision models. Instead of training a custom model, we prompt a multimodal model to return bounding boxes and polygons for the target classes.

## How it works
1) Load an image and define target classes. 2) Build a structured prompt and a strict JSON schema. 3) Send image + prompt to Gemini. 4) Parse and validate the JSON. 5) (Optional) Draw results on the image.

Example JSON schema:
```json
{"detections":[{"class":"person","confidence":0.92,"bbox":[x,y,w,h],"polygon":[[x1,y1],[x2,y2]]}],"image_size":{"width":1280,"height":720}}
```

## Requirements
- Python 3.9+
- Google Generative AI access and API key
- Libraries: google-generativeai, pillow, opencv-python, numpy

Security tip: never hardcode keys. Use environment variables.
- PowerShell: `setx GEMINI_API_KEY "YOUR_KEY"`
- bash: `export GEMINI_API_KEY="YOUR_KEY"`

## Install
```bash
pip install google-generativeai pillow opencv-python numpy
```

## Quickstart (pseudo‑code)
```python
import os, json
import google.generativeai as genai
from PIL import Image

API_KEY = os.getenv("GEMINI_API_KEY"); assert API_KEY, "Set GEMINI_API_KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")  # change to a newer vision model if available

image = Image.open("path/to/image.jpg").convert("RGB")
classes = ["person", "car", "dog"]
schema = {"type":"object","properties":{"detections":{"type":"array","items":{"type":"object","properties":{"class":{"type":"string"},"confidence":{"type":"number"},"bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"polygon":{"type":"array","items":{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2}}},"required":["class","confidence","bbox"]}},"image_size":{"type":"object","properties":{"width":{"type":"number"},"height":{"type":"number"}},"required":["width","height"]}},"required":["detections","image_size"]}

prompt = f"Detect only these classes: {classes}. Return VALID JSON per schema with absolute pixel coords."
resp = model.generate_content([prompt, image], generation_config={
  "response_mime_type": "application/json",
  "response_schema": schema
})
result = json.loads(resp.text)
print(json.dumps(result, indent=2))
```

## Run on Colab / Kaggle
- Click one of the badges above to open the notebook directly in Colab or Kaggle.
- Set the `GEMINI_API_KEY` secret in the environment (Kaggle: Add as a secret; Colab: `os.environ[...]`).

## Repo structure
```
.
├── README.md
├── zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb
└── assets/
    └── pipeline.svg
```

## License & Contributions
Use responsibly and follow Google Generative AI terms. PRs and issues are welcome.