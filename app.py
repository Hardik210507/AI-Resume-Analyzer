import string
import re
from typing import List

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWN_SKILLS = [
    "python", "java", "c", "c++", "javascript", "typescript", "html", "css",
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "excel", "power bi",
    "tableau", "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "machine learning", "deep learning", "data analysis", "data visualization",
    "flask", "django", "streamlit", "fastapi", "git", "github", "docker",
    "aws", "azure", "linux", "api", "rest api", "web scraping", "beautifulsoup",
    "selenium", "nlp", "computer vision", "tensorflow", "pytorch", "react",
    "node.js", "express.js", "communication", "teamwork", "problem solving",
    "leadership", "time management", "critical thinking"
]


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


def get_pdf_text(file_obj) -> str:
    """Extract text from PDF. Basic implementation, scans might fail."""
    try:
        reader = PdfReader(file_obj)
        pages_content = []

        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages_content.append(txt)

        return "\n".join(pages_content).strip()

    except Exception as err:
        st.error(f"Could not read the PDF properly: {err}")
        return ""


def clean_text(raw_input: str) -> str:
    """Normalize text for comparison. Keeps symbols like C++ intact."""
    cleaned = raw_input.lower()

    # Remove standard punctuation but keep + and # for tech skills
    punc = string.punctuation.replace("+", "").replace("#", "")
    cleaned = cleaned.translate(str.maketrans("", "", punc))
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def calc_similarity(resume_content: str, jd_content: str) -> float:
    """TF-IDF cosine similarity between resume and job description."""
    if not resume_content or not jd_content:
        return 0.0

    # Simple TF-IDF is sufficient for this MVP. N-grams can be tuned later.
    vec = TfidfVectorizer(stop_words="english")
    vectors = vec.fit_transform([resume_content, jd_content])

    # Compare first vector (resume) against second (job description)
    sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(sim * 100, 2)


def detect_skills(text: str) -> List[str]:
    """Find which known skills appear in the provided text."""
    clean = clean_text(text)
    found_skills = []

    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        match = re.search(pattern, clean)

        if match:
            found_skills.append(skill.title())

    return sorted(set(found_skills))


def find_missing(resume_skills: List[str], jd_skills: List[str]) -> List[str]:
    """Return skills in the job description that are missing from the resume."""
    resume_set = set()

    for item in resume_skills:
        resume_set.add(item.lower())

    missing = []
    for skill in jd_skills:
        # Using lower() for case-insensitive comparison
        if skill.lower() not in resume_set:
            missing.append(skill)

    return sorted(set(missing))


def calculate_score(sim_score: float, resume_skills: List[str], jd_skills: List[str]) -> int:
    """Generate an ATS-style score based on text similarity and skill overlap."""
    if len(jd_skills) == 0:
        skill_match_score = 0
    else:
        res_set = {skill.lower() for skill in resume_skills}
        jd_set = {skill.lower() for skill in jd_skills}

        matched = res_set.intersection(jd_set)
        skill_match_score = (len(matched) / len(jd_skills)) * 100

    # Weighted average: text similarity is 60%, skills 40%
    final_score = (sim_score * 0.60) + (skill_match_score * 0.40)
    return round(final_score)


def generate_tips(score: int, missing_skills: List[str], resume_text: str) -> List[str]:
    """Create actionable suggestions based on the analysis results."""
    suggestions = []

    if score < 50:
        suggestions.append(
            "Your resume needs stronger alignment with the job description. "
            "Add relevant keywords, tools, and project details where they honestly apply."
        )
    elif score < 75:
        suggestions.append(
            "Your resume is decent, but it could be improved with more role-specific "
            "skills and measurable achievements."
        )
    else:
        suggestions.append("Good match! Your resume is already well aligned with this job description.")

    if missing_skills:
        preview = ", ".join(missing_skills[:8])
        suggestions.append(
            "Consider adding these missing skills if you actually have experience with them: "
            + preview
        )

    return suggestions
st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and paste a job description to check how well they match.")

resume_file = st.file_uploader("Upload your Resume PDF", type=["pdf"])

job_description = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Paste the job description here..."
)

if st.button("Analyze Resume"):
    if resume_file is None:
        st.warning("Please upload your resume PDF.")
    elif not job_description.strip():
        st.warning("Please paste the job description.")
    else:
        resume_text = get_pdf_text(resume_file)

        if not resume_text:
            st.error("Could not extract text from the resume. Try another PDF.")
        else:
            similarity = calc_similarity(resume_text, job_description)

            resume_skills = detect_skills(resume_text)
            jd_skills = detect_skills(job_description)

            missing_skills = find_missing(resume_skills, jd_skills)
            ats_score = calculate_score(similarity, resume_skills, jd_skills)
            tips = generate_tips(ats_score, missing_skills, resume_text)

            st.subheader("📊 Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("ATS Score", f"{ats_score}/100")

            with col2:
                st.metric("Text Similarity", f"{similarity}%")

            st.progress(ats_score / 100)

            st.subheader("✅ Skills Found in Resume")
            if resume_skills:
                st.write(", ".join(resume_skills))
            else:
                st.write("No known skills detected.")

            st.subheader("❌ Missing Skills from Job Description")
            if missing_skills:
                st.write(", ".join(missing_skills))
            else:
                st.write("No major missing skills found.")

            st.subheader("💡 Suggestions")
            for tip in tips:
                st.write("- " + tip)
