# AI Resume Analyzer

A beginner-friendly Python + Streamlit project that analyzes a resume against a job description and gives an ATS-style score, matched skills, missing skills, and improvement suggestions.

## Features

- Upload PDF resume
- Paste job description
- Extract resume text
- Match resume with job description using TF-IDF similarity
- Detect technical skills
- Show matched and missing skills
- Generate ATS-style score
- Give improvement suggestions

## Tech Stack

- Python
- Streamlit
- PyPDF2
- scikit-learn
- pandas

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Folder Structure

```text
ai-resume-analyzer/
├── app.py
├── requirements.txt
└── README.md
```

## Resume Description

AI Resume Analyzer — Built a Python Streamlit web app that analyzes PDF resumes against job descriptions using NLP-based text similarity. Implemented skill extraction, ATS-style scoring, missing keyword detection, and personalized improvement suggestions.
