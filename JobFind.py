import streamlit as st
import requests
import re
import pandas as pd
import io
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import time
import os
import datetime
import concurrent.futures
# from vllm import LLM
# from sentence_transformers import SentenceTransformer


# Inject custom CSS to style buttons within a container with class "big-button"
st.markdown(
    """
    <style>
    div.stButton > button {
        height: 60px !important;   /* Increase button height */
        font-size: 18px !important; /* Increase font size */
        font-weight: bold !important; /* Make text bold */
        padding: 10px !important;  /* Increase padding */
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Define a session state key for evaluated jobs
if "evaluated_jobs" not in st.session_state:
    st.session_state.evaluated_jobs = []

# Global defaults and available fields
DEFAULT_SKILLS = [
    "Python", "Java", "C++", "Machine Learning", "Deep Learning", 
    "Data Analysis", "NLP", "SQL", "JavaScript", "HTML", "CSS", 
    "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy"
]
DEFAULT_FIELDS = ["Job Title", "Employer Name", "Location", "Fit Score", "Explanation","Job Description", "Apply Link"]

# Updated JOB_EVAL_PROMPT_TEMPLATE to include candidate achievements and background.
JOB_EVAL_PROMPT_TEMPLATE = """
You are an expert job recruiter. The candidate has the following profile:
- Key Skills: {key_skills}
- Experience Level: {candidate_experience}
- Achievements: {candidate_achievements}
- Background: {candidate_background}

Evaluate the following job details and determine how good of a fit this job is for the candidate.
Provide a fit score between 0 and 100, where 0 means not a good fit at all and 100 means an excellent fit.
Then provide a brief explanation.
Your response should be in the following format exactly:

Fit Score: <score>
Explanation: <brief explanation>

Job Title: {job_title}
Job Description: {job_description}
"""

# New candidate rating prompt template
CANDIDATE_RATING_PROMPT_TEMPLATE = """
You are an expert recruiter. Please rate the candidate in all relevant areas based on the following resume.
Provide an overall rating between 0 and 100, where 0 means extremely poor and 100 means excellent.
Also provide a brief explanation of the rating.
Resume:
{resume_text}
Rating: <score>
Explanation: <brief explanation>
"""

# Setup constants and initialize models
PDF_STORAGE_PATH = 'pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="llama3.2:latest")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="llama3.2:latest",temperature=0)
# llm_model = LLM(model="deepseek-r1:1.5b")




def extract_candidate_achievements(resume_text):
    """
    Automatically extract candidate achievements from the resume.
    This function searches for sections titled 'Achievements', 'Accomplishments', or 'Awards'
    and returns the text following the section header until the next header or end of text.
    """
    headers = r"(Achievements|Accomplishments|Awards)"
    pattern = re.compile(headers + r"[:\-]*\s*(.*?)(?=\n[A-Z][\w\s]+[:]|$)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(resume_text)
    return match.group(2).strip() if match and match.group(2).strip() else "Not Provided"

def extract_candidate_background(resume_text):
    """
    Automatically extract candidate background from the resume.
    This function searches for sections titled 'Professional Summary', 'Summary', or 'Profile'
    and returns the text following the section header until the next header or end of text.
    """
    headers = r"(Professional Summary|Summary|Profile)"
    pattern = re.compile(headers + r"[:\-]*\s*(.*?)(?=\n[A-Z][\w\s]+[:]|$)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(resume_text)
    return match.group(2).strip() if match and match.group(2).strip() else "Not Provided"

def two_line_summary_no_textwrap(text):
    # First, try to split the text into lines based on newline characters.
    lines = text.splitlines()
    if len(lines) >= 2:
        return "\n".join(lines[:2])
    # If no newlines exist, try splitting by periods to get sentences.
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) >= 2:
        return sentences[0] + ". " + sentences[1] + "."
    # Fallback: return the first 150 characters with ellipsis.
    return text[:150] + "..."


# ------------------------
# Helper Functions
# ------------------------

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def rate_candidate(resume_text):
    """
    Generate an overall rating for the candidate based on their resume.
    Returns a rating score and an explanation.
    """
    rating_prompt = ChatPromptTemplate.from_template(CANDIDATE_RATING_PROMPT_TEMPLATE)
    response_chain = rating_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"resume_text": resume_text})

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def extract_key_skills(resume_text):
    """Extract key skills using simple keyword matching from DEFAULT_SKILLS."""
    found_skills = [skill for skill in DEFAULT_SKILLS if skill.lower() in resume_text.lower()]
    return found_skills

def extract_experience_level(resume_text):
    """
    Extract experience level from the resume using a regex.
    Returns one of: "under_3_years_experience", "more_than_3_years_experience", or "no_experience".
    """
    pattern = re.compile(r'(\d+)\+?\s*(?:years?|years? of experience)', re.IGNORECASE)
    matches = pattern.findall(resume_text)
    if matches:
        years = max([int(match) for match in matches])
        if years < 3:
            return "under_3_years_experience"
        else:
            return "more_than_3_years_experience"
    lower_text = resume_text.lower()
    if "fresher" in lower_text or "entry level" in lower_text or "no experience" in lower_text:
         return "no_experience"
    return "no_experience"

def fetch_jobs(query="developer jobs in ML", job_requirements=None, location_preference=None, country_code=None):
    """
    Fetch jobs from the JSearch API.
    Page and num_pages are fixed at 1.
    The job_requirements parameter is added only if provided.
    If location_preference is "Remote", adds work_from_home=true.
    If "On-site" is chosen and a country code is provided, uses that country code.
    """
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key":  os.getenv("API_KEY", ""),  # Replace 'x' with your actual RapidAPI key.
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
        "Accept": "application/json"
    }
    params = {
        "query": query,
        "page": 1,
        "num_pages": 1,
        "date_posted": "all",
        "language": "",
        "job_requirements":job_requirements
    }
    if job_requirements:
        params["job_requirements"] = job_requirements
        print("Job requirements: ",job_requirements)
    if location_preference == "Remote":
        params["work_from_home"] = "true"
        if "country" in params:
            del params["country"]
    elif location_preference == "On-site" and country_code:
        params["country"] = country_code
    response = requests.get(url, headers=headers, params=params)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch job listings."}

def evaluate_job_fit(
    key_skills,
    candidate_experience="Not Provided",
    candidate_achievements="Not Provided",
    candidate_background="Not Provided",
    job=None
):
    """
    Evaluate job fit using the model, considering the candidate's skills, experience,
    achievements, and background. The achievements and background parameters are optional.
    
    Returns the evaluation string.
    """
    if job is None:
        job = {}
    job_eval_prompt = ChatPromptTemplate.from_template(JOB_EVAL_PROMPT_TEMPLATE)
    response_chain = job_eval_prompt | LANGUAGE_MODEL
    evaluation = response_chain.invoke({
        "key_skills": ", ".join(key_skills),
        "candidate_experience": candidate_experience,
        "candidate_achievements": candidate_achievements if candidate_achievements else "Not Provided",
        "candidate_background": candidate_background if candidate_background else "Not Provided",
        "job_title": job.get("job_title", "No Title"),
        "job_description": job.get("job_description", "No description provided.")
    })
    return evaluation

def generate_chat_response(user_message):
    """
    Default chat response (without document context).
    """
    chat_prompt_template = "You are a helpful assistant.\nUser: {user_message}\nAssistant:"
    chat_prompt = ChatPromptTemplate.from_template(chat_prompt_template)
    response_chain = chat_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_message": user_message})

def generate_chat_response_with_doc(user_message, resume_text):
    """
    Generate a chat response using the uploaded document (resume) as context.
    """
    chat_prompt_template = """You are a helpful assistant. The following is the candidate's resume:
{resume_text}

Answer the user's query based on the resume.
User: {user_message}
Assistant:"""
    chat_prompt = ChatPromptTemplate.from_template(chat_prompt_template)
    response_chain = chat_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"resume_text": resume_text, "user_message": user_message})

def generate_recommendation(additional_info):
    """
    Generate a recommendation (e.g., additional skill or query modification)
    based on the candidate's additional information.
    """
    rec_prompt_template = """You are a job advisor. The candidate provided the following additional information:
{info}

Based on this, recommend one or two additional skills or suggest a modification to the job query that could improve the job search results.
Recommendation:"""
    rec_prompt = ChatPromptTemplate.from_template(rec_prompt_template)
    rec_chain = rec_prompt | LANGUAGE_MODEL
    return rec_chain.invoke({"info": additional_info})

# ------------------------
# Tabbed Interface
# ------------------------

tabs = st.tabs(["Job Search", "Chat with AI Recruiter", "Job Details Config"])
job_search_tab, chat_tab, field_options_tab = tabs

# ------------------------
# FIELD OPTIONS TAB
# ------------------------

import json

def get_config():
    """
    Collects the current configuration from st.session_state.
    Adjust the keys as needed.
    """
    config = {
        "candidate_skills": st.session_state.get("candidate_skills", []),
        "ranked_skills": st.session_state.get("ranked_skills", []),
        "candidate_experience": st.session_state.get("candidate_experience", ""),
        "candidate_achievements": st.session_state.get("candidate_achievements", ""),
        "candidate_background": st.session_state.get("candidate_background", ""),
        "working_preference": st.session_state.get("working_preference", ""),
        "country_code": st.session_state.get("country_code", ""),
        "additional_info": st.session_state.get("additional_info", ""),
        "job_query": st.session_state.get("job_query", ""),
        "selected_fields": st.session_state.get("selected_fields", DEFAULT_FIELDS)
    }
    return config

def save_config_to_json():
    """
    Returns the configuration as a JSON string.
    """
    config = get_config()
    return json.dumps(config, indent=2)

def load_config_from_json(config_json):
    """
    Loads the configuration from a JSON string and updates st.session_state.
    """
    try:
        config = json.loads(config_json)
        for key, value in config.items():
            st.session_state[key] = value
    except Exception as e:
        st.error("Failed to load config: " + str(e))

with field_options_tab:
    st.title("🛠 Job Details Config")
    selected_fields = st.multiselect(
        "Select Fields to Display in Job Listings and Excel",
        options=DEFAULT_FIELDS,
        default=DEFAULT_FIELDS,
        key="selected_fields"
    )
    # st.session_state["selected_fields"] = selected_fields

    st.markdown("### Save / Load Configuration")
    config_json = save_config_to_json()
    st.download_button(
        label="Download Config as JSON",
        data=config_json,
        file_name="config.json",
        mime="application/json"
    ,
    use_container_width=True)

    config_file = st.file_uploader("Upload Config JSON", type="json", key="config_uploader")
    if config_file:
        config_json_loaded = config_file.read().decode("utf-8")
        load_config_from_json(config_json_loaded)
        st.success("Configuration loaded successfully!")

# ------------------------
# JOB SEARCH TAB
# ------------------------

with job_search_tab:
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.title("🔎 JobFind")
    st.write("")
    st.markdown("##### Personalized Jobs delivered within a minute ⌚")
    st.write("")
    st.write("")
    st.markdown("### Resume Upload")
    st.write("")
    
    # Upload Resume Section
    uploaded_pdf = st.file_uploader(
        "",
        type="pdf",
        help="Select a PDF resume for analysis",
        accept_multiple_files=False
    )
    
    if uploaded_pdf:
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        resume_text = raw_docs[0].page_content
        st.session_state["uploaded_resume_text"] = resume_text

        # Auto-extract candidate details
        extracted_skills = extract_key_skills(resume_text)
        extracted_experience = extract_experience_level(resume_text)
        extracted_achievements = extract_candidate_achievements(resume_text)
        extracted_background = extract_candidate_background(resume_text)

        # Achievements & Background (Hidden by default in an expandable section)
        with st.expander("Extracted Information", expanded=False):
            candidate_skills = st.multiselect(
            "Skills",
            options=DEFAULT_SKILLS,
            default=extracted_skills
            )
            ranked_skills_str = st.text_area(
                "Top 3 skills from best to worst (, separated).",
                value=", ".join(candidate_skills[:3])
            )
            candidate_achievements = st.text_area("Candidate Achievements (Optional)", value=extracted_achievements)
            candidate_background = st.text_area("Candidate Background (Optional)", value=extracted_background)

        # Advanced Settings: Hide additional details for a cleaner UI
        with st.expander("Advanced Settings", expanded=False):
            experience_options = ["Any", "no_experience", "under_3_years_experience", "more_than_3_years_experience"]
            candidate_experience = st.selectbox(
                "Select Experience Level (optional)",
                options=experience_options,
                index=0
            )
            location_options = ["Any", "Remote", "On-site"]
            working_preference = st.selectbox("Select Working Preference", options=location_options, index=0)
            country_code = ""
            if working_preference == "On-site":
                country_code = st.text_input("Enter Country Code for On-site jobs", value="us")
            additional_info = st.text_area(
                "Additional Information (Optional)",
                "Enter any extra details you want us to know (e.g., interests, domain background, etc.)"
            )
            # sort_by_fit = st.checkbox("Sort jobs by fit score", value=False)

        ranked_skills = extracted_skills[:3]

        # Default values if settings are not expanded
        if "working_preference" not in locals():
            working_preference = "Any"
        if "additional_info" not in locals():
            additional_info = ""
        
        # Build job query based on basic inputs
        default_query = "Jobs in " + " ".join(ranked_skills)
        # if candidate_experience != "Any":
        #     default_query += " " + candidate_experience
        if working_preference == "Remote":
            default_query += " remote"
        elif working_preference == "On-site" and country_code:
            default_query += " " + country_code
        st.write("")
        st.write("")
        job_query = st.text_area(" Job Query (please put 3 skills at a time)", value=default_query)
        st.write("")
        st.write("")
    
        if st.button("Search Jobs", use_container_width=True):
                # Start overall search timer
                search_start = time.time()
                job_req = candidate_experience if candidate_experience != "Any" else None
                loc_pref = working_preference if working_preference != "Any" else None
                col1,col2,col3 = st.columns([1.6,2,1])
                with col2:
                    with st.spinner("Fetching Jobs..."):
                        api_results_start = time.time()
                        job_results = fetch_jobs(query=job_query, job_requirements=job_req, location_preference=loc_pref, country_code=country_code)
                        api_results_end = time.time()
                        api_results_processing_time = api_results_end - api_results_start
                        # with col3:
                        st.write(f"*Time taken: {api_results_processing_time:.2f} seconds*")
                        st.markdown("---")

                if "error" in job_results:
                    st.error(job_results["error"])
                else:
                    col1,col2,col3 = st.columns([1.6,2,1])
                    with col2:
                        st.markdown("""
                            <script>
                                document.getElementById('job-listings').scrollIntoView({ behavior: 'smooth' });
                            </script>
                        """, unsafe_allow_html=True)
                        st.write("## Job Listings:")
                    st.markdown("""
                        <script>
                            document.getElementById('job-listings').scrollIntoView({ behavior: 'smooth' });
                        </script>
                    """, unsafe_allow_html=True)
                    evaluated_jobs = []
                    jobs = job_results.get("data", [])
                    # Create a ThreadPoolExecutor to process explanations concurrently.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        future_results = []  # This will store tuples with both futures and placeholders.
                        
                        # Loop through jobs and render static details.
                        for idx, job in enumerate(jobs):
                            job_start = time.time()
                            selected_fields = st.session_state.get("selected_fields", DEFAULT_FIELDS)
                            
                            # Display static job details.
                            if "Job Title" in selected_fields:
                                st.write(f"#### {job.get('job_title', 'Unknown Title')}")
                            if "Employer Name" in selected_fields:
                                st.write(f"##### {job.get('employer_name', 'N/A')}")
                            
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col3:
                                if "Fit Score" in selected_fields:
                                    fitscore_placeholder = st.empty()
                                    fitscore_placeholder.write("Fit Score: Pending")
                            with col1:
                                if "Location" in selected_fields:
                                    st.write(f"Location: {job.get('job_location', 'N/A')}")
                            
                            if "Job Description" in selected_fields:
                                # Instead of processing summary synchronously, create a placeholder.
                                st.write("##### **Job Description Summary:**")
                                summary_placeholder = st.empty()
                                summary_placeholder.write("Loading summary...")
                            else:
                                summary_placeholder = None

                            if "Job Description" in selected_fields:
                                with st.expander("Job Description"):
                                    st.write(job.get("job_description", "No description provided."))
                            
                            # Create a placeholder for the explanation.
                            if "Explanation" in selected_fields:
                                expander_explanation = st.expander("Explanation", expanded=False)

                                exp_placeholder = expander_explanation.empty()
                                exp_placeholder.write("Explanation: Loading...")
                            else:
                                exp_placeholder = None

                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                if "Apply Link" in selected_fields:
                                    apply_link = job.get("job_apply_link", "#")
                                    st.link_button("Apply Here", apply_link, use_container_width=True)
                            with col3:
                                time_placeholder = st.empty()
                            st.markdown("---")
                            
                            # Submit the evaluation and summary concurrently.
                            eval_future = executor.submit(
                                evaluate_job_fit,
                                ranked_skills,
                                candidate_experience,
                                candidate_achievements,
                                candidate_background,
                                job
                            )
                            summary_future = executor.submit(
                                two_line_summary_no_textwrap,
                                job.get("job_description", "No description provided.")
                            )
                            
                            future_results.append((eval_future, summary_future, exp_placeholder, summary_placeholder,fitscore_placeholder,  time_placeholder, job_start, job))
                        
                        # Process completed futures in the main thread.
                        for eval_future, summary_future, exp_placeholder, summary_placeholder,fitscore_placeholder, time_placeholder, job_start, job in future_results:
                            # Process explanation.
                            try:
                                evaluation = eval_future.result(timeout=30)
                            except concurrent.futures.TimeoutError:
                                evaluation = "skipped"
                            except Exception:
                                evaluation = "skipped"
                            
                            try:
                                score_match = re.search(r"Fit Score:\s*(\d+)", evaluation)
                                score = int(score_match.group(1)) if score_match else 0
                                explanation_match = re.search(r"Explanation:\s*(.*)", evaluation)
                                explanation = explanation_match.group(1).strip() if explanation_match else evaluation
                            except Exception:
                                score, explanation = 0, evaluation
                            
                            # Process summary.
                            try:
                                summary = summary_future.result(timeout=30)
                            except concurrent.futures.TimeoutError:
                                summary = "skipped"
                            except Exception:
                                summary = "skipped"
                            
                            # Update placeholders in the main thread.
                            if exp_placeholder is not None:
                                with exp_placeholder:
                                    st.write(f"Explanation: {explanation}")
                            if summary_placeholder is not None:
                                with summary_placeholder:
                                    st.write(summary)
                            if fitscore_placeholder is not None:
                                    with fitscore_placeholder:
                                        st.write(f"**Fit Score:** {score}")
                            
                            
                            job_end = time.time()
                            job_processing_time = job_end - job_start
                            col1,col2,col3 = st.columns([1,1,1])
                            with col3:
                                with time_placeholder:
                                    st.write(f"*Time taken: {job_processing_time:.2f} seconds*")
                            
                            evaluated_jobs.append((score, job, explanation))
                                            
                    st.session_state.evaluated_jobs = evaluated_jobs
                    # Clear previous display before updating

                    if evaluated_jobs:
                        export_data = []
                        for score_val, job_item, expl in evaluated_jobs:
                            export_data.append({
                                "Job Title": job_item.get('job_title', 'Unknown Title'),
                                "Employer Name": job_item.get('employer_name', 'N/A'),
                                "Location": job_item.get('job_location', 'N/A'),
                                "Fit Score": score_val,
                                "Explanation": expl,
                                "Job Description": job_item.get('job_description', 'No description provided.'),
                                "Apply Link": job_item.get('job_apply_link', '#')
                            })
                        df_jobs = pd.DataFrame(export_data)
                        towrite = io.BytesIO()
                        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                            df_jobs.to_excel(writer, index=False, sheet_name='Job Listings')
                        towrite.seek(0)
                        st.download_button(
                        label="Download as Excel",
                        data=towrite,
                        file_name="job_listings.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                    overall_end = time.time()
                    overall_time = overall_end - search_start
                    col1,col2,col3 = st.columns([1,1,1])
                    with col2:
                        st.markdown(f"<p style='font-size:0.8em;'>Total time taken: {overall_time:.2f} seconds</p>", unsafe_allow_html=True)


                    # Prepare candidate info (one row DataFrame)
                    resume_filename = uploaded_pdf.name
                    file_basename = os.path.splitext(resume_filename)[0]
                    candidate_info = {
                        "Resume Filename": resume_filename,
                        "Candidate Skills": ", ".join(candidate_skills),
                        "Ranked Skills": ", ".join(ranked_skills),
                        "Candidate Achievements": candidate_achievements,
                        "Candidate Background": candidate_background,
                        "Candidate Experience": candidate_experience,
                        "Working Preference": working_preference,
                        "Country Code": country_code,
                        "Additional Info": additional_info,
                        "Job Query": job_query,
                        "API Fetch Time (sec)": f"{api_results_processing_time:.2f}",
                        "Total Search Time (sec)": f"{overall_time:.2f}",
                        "Timestamp": datetime.datetime.now().isoformat()
                    }
                    df_candidate = pd.DataFrame([candidate_info])

                    # Create an Excel file with both sheets.
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"jobs/{file_basename}-{timestamp_str}.xlsx"
                    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                        df_candidate.to_excel(writer, index=False, sheet_name="Candidate Info")
                        if not df_jobs.empty:
                            df_jobs.to_excel(writer, index=False, sheet_name="Job Listings")
                    
# ------------------------
# CHAT WITH THE RECRUITER TAB
# ------------------------


# Create tabs (assuming you might have other tabs as well)



with chat_tab:
    st.title("💬 Chat with AI Recruiter")
    st.markdown(
        "Ask questions about your uploaded resume or request additional candidate feedback "
        "or get your resume reviewed by AI in seconds."
    )

    # Layout columns for buttons.
    col1, col2, col3 = st.columns([2, 4, 1])
    with col3:
        if st.button("Clear"):
            st.session_state["chat_history"] = []
    with col1:
        if st.button("Rate My Resume"):
            resume_context = st.session_state.get("uploaded_resume_text", "")
            if resume_context:
                rating_response = rate_candidate(resume_context)
                st.session_state.setdefault("chat_history", []).append(
                    ("Recruiter (Rating)", rating_response)
                )
            else:
                st.warning("Please upload your Resume")

    # Retrieve the resume context (if any)
    resume_context = st.session_state.get("uploaded_resume_text", "")

    # Ensure chat history exists.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Create an empty container for chat history.
    chat_history_container = st.empty()

    # Function to update the chat container.
    def update_chat_history():
        chat_display = ""
        for role, message in st.session_state["chat_history"]:
            chat_display += f"**{role}:** {message}\n\n---\n\n"
        chat_history_container.markdown(chat_display)
    
    # Initial display of chat history.
    update_chat_history()

    # Chat input form (clears input on submit).
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your question")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            # Append user's message.
            st.session_state["chat_history"].append(("User", user_input))
            
            with st.spinner("Generating response..."):
                if resume_context:
                    assistant_response = generate_chat_response_with_doc(user_input, resume_context)
                else:
                    assistant_response = generate_chat_response(user_input)
            
            # Append assistant's response.
            st.session_state["chat_history"].append(("Recruiter", assistant_response))
            
            # Update chat history immediately.
            update_chat_history()

    # Auto-scroll to bottom (if needed).
    st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <script>
            var bottom = document.getElementById('bottom');
            if(bottom) {
                bottom.scrollIntoView({ behavior: 'smooth' });
            }
        </script>
        """,
        unsafe_allow_html=True
    )
st.markdown("""
    <footer style="position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 0.8em; padding: 10px 0;">
      
      Created by Aman Ali 2025
    </footer>
    """, unsafe_allow_html=True)
