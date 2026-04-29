import re
import string
from typing import List, Tuple

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILLS = [
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


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as error:
        st.error(f"Could not read PDF: {error}")
        return ""


def clean_text(text: str) -> str:
    """Lowercase and remove extra punctuation/spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("+", "").replace("#", "")))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_similarity(resume_text: str, job_description: str) -> float:
    """Calculate cosine similarity between resume and job description."""
    if not resume_text or not job_description:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)


def extract_skills(text: str) -> List[str]:
    """Find known skills present in text."""
    cleaned = clean_text(text)
    found_skills = []

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, cleaned):
            found_skills.append(skill.title())

    return sorted(set(found_skills))


def get_missing_skills(resume_skills: List[str], jd_skills: List[str]) -> List[str]:
    """Find job description skills missing from resume."""
    resume_set = {skill.lower() for skill in resume_skills}
    missing = [skill for skill in jd_skills if skill.lower() not in resume_set]
    return sorted(set(missing))


def generate_score(similarity_score: float, resume_skills: List[str], jd_skills: List[str]) -> int:
    """Generate ATS score using text similarity and skill match ratio."""
    if not jd_skills:
        skill_score = 0
    else:
        matched = len(set(skill.lower() for skill in resume_skills) & set(skill.lower() for skill in jd_skills))
        skill_score = (matched / len(jd_skills)) * 100

    final_score = (similarity_score * 0.6) + (skill_score * 0.4)
    return round(final_score)


def generate_suggestions(ats_score: int, missing_skills: List[str], resume_text: str) -> List[str]:
    """Generate simple improvement suggestions."""
    suggestions = []

    if ats_score < 50:
        suggestions.append("Your resume needs stronger alignment with the job description. Add more relevant keywords and project details.")
    elif ats_score < 75:
        suggestions.append("Your resume is decent, but it can improve by adding more role-specific skills and measurable achievements.")
    else:
        suggestions.append("Good match! Your resume is well aligned with the job description.")

    if missing_skills:
        suggestions.append("Consider adding these missing skills if you have experience with them: " + ", ".join(missing_skills[:8]) + ".")

    if len(resume_text.split()) < 250:
        suggestions.append("Your resume text seems short. Add more detail about projects, internships, certifications, and achievements.")

    if not re.search(r"\d+", resume_text):
        suggestions.append("Add measurable achievements, such as percentages, numbers, rankings, or project impact.")

    if "github" not in resume_text.lower() and "linkedin" not in resume_text.lower():
        suggestions.append("Add your GitHub and LinkedIn links to make your profile stronger.")

    return suggestions


def display_skill_table(title: str, skills: List[str]):
    st.subheader(title)
    if skills:
        skill_df = pd.DataFrame({"Skills": skills})
        st.dataframe(skill_df, use_container_width=True, hide_index=True)
    else:
        st.info("No skills found.")


st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and paste a job description to get an ATS-style score and improvement suggestions.")

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("Upload Resume PDF", type=["pdf"])

with col2:
    job_description = st.text_area("Paste Job Description", height=250)

analyze_button = st.button("Analyze Resume", type="primary")

if analyze_button:
    if uploaded_resume is None:
        st.warning("Please upload a resume PDF.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Analyzing resume..."):
            resume_text = extract_text_from_pdf(uploaded_resume)

            if not resume_text:
                st.error("No readable text found in the PDF. Try uploading a text-based PDF instead of a scanned image.")
            else:
                similarity_score = calculate_similarity(resume_text, job_description)
                resume_skills = extract_skills(resume_text)
                jd_skills = extract_skills(job_description)
                missing_skills = get_missing_skills(resume_skills, jd_skills)
                ats_score = generate_score(similarity_score, resume_skills, jd_skills)
                suggestions = generate_suggestions(ats_score, missing_skills, resume_text)

                st.success("Analysis completed!")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("ATS Score", f"{ats_score}/100")
                metric_col2.metric("Text Similarity", f"{similarity_score}%")
                metric_col3.metric("Skills Found", len(resume_skills))

                st.progress(ats_score / 100)

                left, right = st.columns(2)
                with left:
                    display_skill_table("✅ Skills Found in Resume", resume_skills)
                with right:
                    display_skill_table("🎯 Skills Required in Job Description", jd_skills)

                display_skill_table("⚠️ Missing Skills", missing_skills)

                st.subheader("💡 Suggestions")
                for suggestion in suggestions:
                    st.write("- " + suggestion)

                with st.expander("View Extracted Resume Text"):
                    st.write(resume_text)

st.sidebar.title("About")
st.sidebar.info(
    "This project uses Python, Streamlit, PyPDF2, TF-IDF, and cosine similarity "
    "to compare a resume with a job description."
)

st.sidebar.title("Tips")
st.sidebar.write("- Use a text-based PDF resume.")
st.sidebar.write("- Paste a complete job description.")
st.sidebar.write("- Add missing skills only if you genuinely know them.")