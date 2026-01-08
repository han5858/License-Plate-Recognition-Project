# 🚗 AI License Plate Recognition System (LPR)

![Demo Animation](gif.gif)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-green?style=for-the-badge)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Text_Recognition-yellow?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge)

## 📖 Overview
This project is a real-time **License Plate Recognition (LPR)** system built with Python. It detects license plates from video footage using **YOLOv8**, extracts the text using **EasyOCR**, and verifies the license plate against a database using **Levenshtein Distance** (Fuzzy Logic).

It simulates a **Smart Gate System** that automatically grants or denies access based on the license plate.

---

## ✨ Key Features
* **🔍 High Accuracy Detection:** Uses a custom-trained YOLOv8 model (`plate_model.pt`) to detect license plates.
* **📝 Optical Character Recognition (OCR):** Extracts text from the detected region using EasyOCR.
* **🧠 Fuzzy Matching Algorithm:** Uses **Levenshtein Distance** to correct minor OCR errors (e.g., reading '0' as 'O') and matches plates with >70% similarity.
* **📊 Data Logging:** Automatically saves all detected plates, timestamps, and status to an **Excel Report (`plate_report.xlsx`)**.
* **📹 Visual Output:** Generates a processed video with bounding boxes and status labels (Green = Authorized, Red = Unknown).

---

## 🛠️ Tech Stack & Libraries
* **Ultralytics YOLOv8:** For object detection (License Plate localization).
* **EasyOCR:** For reading text from images.
* **OpenCV:** For video processing and visualization.
* **Levenshtein:** For calculating string similarity ratio.
* **Pandas:** For exporting data to Excel.

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/AI-License-Plate-Recognition.git](https://github.com/YOUR_USERNAME/AI-License-Plate-Recognition.git)
cd AI-License-Plate-Recognition
