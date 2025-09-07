# Zero‑Shot Object Detection & Segmentation with Google Gemini

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aminhaghii/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5/blob/main/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb)
[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/aminhaghii/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5/main/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb)

![Pipeline](assets/pipeline.svg)

## Table of Contents
- Project Overview
- Features & Benefits
- Architecture & Workflow
- Setup & Installation
- Usage (Notebook and Script)
- Colab & Kaggle
- Troubleshooting
- Contributions
- License

## Project Overview
This repository demonstrates a practical, end‑to‑end approach to perform zero‑shot object detection and segmentation using Google Gemini Vision models. You do not need a labeled dataset or model fine‑tuning. Instead, you construct a structured prompt and ask the multimodal model to return detections (bounding boxes) and, optionally, segmentation polygons for specified target classes. Typical use cases include rapid prototyping, exploring new label taxonomies, or augmenting existing pipelines without retraining.

Scope and Key Functionalities:
- Accepts an input image and a list of target classes.
- Builds a strict JSON schema for outputs to minimize parsing errors.
- Sends image + prompt to the Gemini Vision model and receives structured output.
- Validates and parses the model’s JSON, then optionally visualizes results.

## Features & Benefits
- Zero-shot: No labeled data or training required.
- Unified detection + segmentation: bbox and optional polygon per instance.
- Flexible: Works for arbitrary class lists at inference time.
- Portable: Runs via a single notebook or short Python snippet.
- Secure by design: Uses environment variables for API keys.
- Cloud-friendly: One-click launch in Colab and Kaggle.

## Architecture & Workflow
High-level flow:
1) Input image + target classes
2) Structured prompt + strict JSON schema
3) Gemini Vision inference (image + prompt)
4) JSON validation & parsing
5) Optional visualization and export to files

The included diagram (pipeline.svg) depicts the end-to-end path from inputs to outputs.

Example JSON schema (truncated for brevity):
```json
{"detections":[{"class":"person","confidence":0.92,"bbox":[x,y,w,h],"polygon":[[x1,y1],[x2,y2]]}],"image_size":{"width":1280,"height":720}}
```

## Setup & Installation
Requirements:
- Python 3.9+
- Access to Google Generative AI and an API key
- Libraries: google-generativeai, pillow, opencv-python, numpy

Install dependencies:
```bash
pip install google-generativeai pillow opencv-python numpy
```

Set your API key securely (never hardcode in code):
- PowerShell (Windows):
  - setx GEMINI_API_KEY "YOUR_KEY"
- bash (Linux/macOS):
  - export GEMINI_API_KEY="YOUR_KEY"
- Colab:
  - In a cell: import os; os.environ["GEMINI_API_KEY"] = "YOUR_KEY"
- Kaggle:
  - Add GEMINI_API_KEY as a secret in the notebook settings and access via os.environ["GEMINI_API_KEY"].

## Usage (Notebook and Script)
Recommended: use the notebook for an end-to-end run:
- Open in Colab or Kaggle using the badges at the top.
- Ensure GEMINI_API_KEY is set.
- Run the cells in order.

Minimal Python example:
```python
import os, json
import google.generativeai as genai
from PIL import Image

API_KEY = os.getenv("GEMINI_API_KEY"); assert API_KEY, "Set GEMINI_API_KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")  # update to your available vision model

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

Optional visualization with OpenCV (draw bboxes and polygons):
```python
import cv2
import numpy as np

img = cv2.imread("path/to/image.jpg")
for det in result.get("detections", []):
    x, y, w, h = map(int, det["bbox"])  # [x, y, width, height]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if "polygon" in det and det["polygon"]:
        poly = np.array(det["polygon"], dtype=np.int32)
        cv2.polylines(img, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
cv2.imwrite("output_vis.jpg", img)
```

Notes:
- Always set response_mime_type and response_schema to reduce JSON parsing errors.
- If a newer Gemini Vision model is available to you, update the model name accordingly.
- Consider resizing very large images to meet request limits and speed up inference.

## Colab & Kaggle
- [Open in Colab](https://colab.research.google.com/github/aminhaghii/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5/blob/main/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb)
- [Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/aminhaghii/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5/main/zero_shot_object_detection_and_segmentation_with_google_gamini_2_5.ipynb)

These buttons are also available at the top of this README as badges for one-click access.

## Troubleshooting
- Authentication/401 or 403 errors: Check GEMINI_API_KEY is set and valid; verify model access.
- Non-JSON or malformed responses: Ensure response_mime_type and response_schema are provided; re‑prompt with clearer constraints.
- Rate limits/timeouts: Retry with backoff; reduce image size; avoid overly long prompts.
- Visualization issues: Verify bbox format [x, y, width, height] and polygon points [[x1,y1], ...].

## Contributions
Contributions are welcome! Suggested process:
- Fork the repository and create a feature branch.
- Make focused changes with clear commit messages.
- Add or update examples if applicable.
- Open a pull request describing your changes and testing steps.

## License
No explicit LICENSE file is currently provided in this repository. If you plan to reuse or distribute this project, please add a license file (e.g., MIT, Apache‑2.0, or GPL‑3.0). Regardless of your choice, follow Google Generative AI usage policies for API access and content.