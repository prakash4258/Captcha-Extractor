**CAPTCHA Solver**


A Python project for automatically solving distorted and noisy CAPTCHAs using advanced image pre-processing and the EasyOCR deep learning engine.

This tool is designed to achieve high accuracy by cleaning up challenging CAPTCHA images before they are passed to the OCR engine.
````markdown
# CAPTCHA Solver with EasyOCR & OpenCV

A Python project to extract text from noisy CAPTCHA images using **EasyOCR** and **OpenCV**.

## Features
- Preprocesses CAPTCHA images (noise removal, binarization, thresholding)
- Uses EasyOCR for text extraction
- Handles noisy and distorted images
- Supports CPU and GPU (GPU is faster)

## Requirements
- Python 3.8+
- Install dependencies:

```bash
pip install easyocr opencv-python torch torchvision
````

## Usage

1. Place your CAPTCHA image in the project folder.
2. Update the image path in `captcha_solver.py` if needed.
3. Run the script:

```bash
python captcha_solver.py
```

4. The script will:

   * Preprocess the image
   * Save a cleaned version as `input.png ` and `seg_clean.png`
   * Display the detected text

## Example

**Input CAPTCHA:**
![Example CAPTCHA](captcha.png)

**Output:**

```
Extracted Text: 86218
```

## Project Structure

```
.
├── README.md             # Documentation
├── captcha.png           #input image
├── captcha_solver.py     # Main script
├── input.png             # Sample CAPTCHA image
├── seg_clean.png         # Preprocessed image

```

## License

MIT License

```

Do you also want me to include a **section in the README for GitHub hosting instructions** so people know how to clone and run it? That would make it easier for others to use.
```
