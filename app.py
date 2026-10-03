import os
import tempfile
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

MODEL = "gemini-3.6-flash"

st.title("📄 AI Resume ATS Analyzer")
st.caption(
    "Upload a resume to get an estimated ATS-compatibility score and practical improvement suggestions."
)

st.info(
    "Note: The score is an AI-based estimate, not the score produced by a specific employer's ATS. "
    "Actual ATS behavior varies by system and job posting."
)

# -----------------------------
# Gemini client
# -----------------------------
api_key = None

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning(
        "Gemini API key is not configured. Add GEMINI_API_KEY in Streamlit Cloud "
        "Secrets or as an environment variable."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Job Target")
    job_description = st.text_area(
        "Optional job description",
        height=220,
        placeholder="Paste the job description here for a more targeted ATS analysis.",
    )

    st.markdown("---")
    st.markdown(
        "**Supported file:** PDF\n\n"
        "**Recommended:** 1–5 page text-based resume"
    )

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"],
    help="Upload a PDF resume. Avoid password-protected files.",
)

def analyze_resume(pdf_path: str, job_description: str) -> str:
    """Send the PDF resume and analysis prompt to Gemini."""
    prompt = f"""
You are an expert resume reviewer and ATS-optimization assistant.

Analyze the uploaded resume carefully.

Your task:
1. Give an estimated ATS compatibility score from 0 to 100.
2. Explain the score using concrete evidence from the resume.
3. Identify ATS risks such as:
   - missing or weak keywords
   - unclear section headings
   - poor formatting or structure
   - missing contact information
   - weak action verbs
   - vague or non-quantified achievements
   - skills that are difficult for ATS parsing
   - excessive graphics/tables/text boxes if visible
4. Identify strengths.
5. Give specific improvements, prioritized from highest impact to lowest impact.
6. Suggest improved wording for up to 5 weak resume bullets, but do not invent facts,
   achievements, employers, dates, skills, metrics, or credentials.
7. If a job description is provided, compare the resume against it and identify:
   - important matching keywords
   - important missing keywords/skills
   - areas where the resume could better demonstrate the stated requirements.

IMPORTANT:
- Do not claim that your score is the actual score from an ATS.
- Base the analysis only on information visible in the resume and the supplied job description.
- Do not invent missing information.
- Keep the advice practical and concise.

JOB DESCRIPTION:
{job_description if job_description.strip() else "No job description provided. Evaluate general ATS compatibility."}

Return the answer in exactly this structure:

# ATS Compatibility Score
**Score: XX/100**

## Score Breakdown
- Keyword alignment: X/20
- Resume structure & ATS readability: X/20
- Skills relevance: X/20
- Experience/achievement quality: X/20
- Completeness & clarity: X/20

## Strengths
- ...

## ATS Issues
- ...

## Priority Improvements
1. ...
2. ...
3. ...

## Suggested Bullet Improvements
- Original: ...
  Improved: ...

## Keyword Analysis
### Matching Keywords
- ...

### Missing / Recommended Keywords
- ...

## Final Checklist
- [ ] ...
- [ ] ...
"""

    uploaded = client.files.upload(file=pdf_path)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            uploaded,
            types.Part.from_text(text=prompt),
        ],
    )

    return response.text


# -----------------------------
# Analyze button
# -----------------------------
if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload a PDF resume first.")
    else:
        try:
            suffix = Path(uploaded_file.name).suffix.lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            with st.spinner("Analyzing your resume with Gemini..."):
                result = analyze_resume(temp_path, job_description)

            st.success("Analysis complete.")
            st.markdown(result)

        except Exception as exc:
            st.error("The resume could not be analyzed.")
            st.exception(exc)

        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

st.markdown("---")
st.caption("AI Resume ATS Analyzer • Gemini 2.5 Flash • Streamlit")
