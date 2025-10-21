import streamlit as st
import os
import io

# --- Import SemanticPlagiarismDetector class from detector.py ---
from detector import SemanticPlagiarismDetector

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Semantic Plagiarism Detector",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding: 2rem;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    h1 { color: #1a1a2e; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h2, h3 { color: #2e2e4f; }
    .stButton > button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; font-size: 1.1rem; font-weight: bold; }
    .stButton > button:hover { background-color: #45a049; }
    .stTextArea > label, .stFileUploader > label { font-weight: bold; color: #2e2e4f; }
    mark.plagiarized { background-color: #fff700; color: #c70000; padding: 2px 3px; border-radius: 3px; font-weight: bold; }
    .serpapi-status { padding: 10px; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #333; }
    .serpapi-status.active { background-color: #e8f5e9; border: 1px solid #4CAF50; }
    .serpapi-status.default-key { background-color: #fffde7; border: 1px solid #ffeb3b; }
    .serpapi-status.not-set { background-color: #ffebee; border: 1px solid #f44336; }
</style>
""", unsafe_allow_html=True)

# --- Instantiate detector ---
@st.cache_resource
def get_plagiarism_detector_instance():
    serpapi_key = None

    if hasattr(st, 'secrets'):
        serpapi_key = st.secrets.get("SERPAPI_KEY")
    if not serpapi_key:
        serpapi_key = os.environ.get("SERPAPI_KEY")

    DEFAULT_SERPAPI_KEY = "Your_serp_api_key"
    if not serpapi_key:
        serpapi_key = DEFAULT_SERPAPI_KEY

    detector_instance = SemanticPlagiarismDetector(serpapi_key=serpapi_key)

    if serpapi_key == DEFAULT_SERPAPI_KEY and \
       (not (hasattr(st, 'secrets') and st.secrets.get("SERPAPI_KEY") == serpapi_key) and \
        not (os.environ.get("SERPAPI_KEY") == serpapi_key)):
        st.warning(
            "⚠️ SerpAPI key is being used from a default/hardcoded value. "
            "Set your key in `.streamlit/secrets.toml` or as an environment variable (`SERPAPI_KEY`)."
        )
    return detector_instance, serpapi_key == DEFAULT_SERPAPI_KEY

detector, using_default_serpapi_key = get_plagiarism_detector_instance()

# --- Header ---
st.title("📚 AI Semantic Plagiarism Detector")
st.markdown("Analyze text for semantic similarity against web sources and internal documents.")
st.markdown("---")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Plagiarism Threshold", 0.60, 0.99, 0.80, 0.01)
    num_rephrase_suggestions = st.slider("Number of Rephrasing Suggestions", 1, 5, 3)
    st.markdown("---")
    st.subheader("SerpAPI Status")
    if detector.serpapi_key and detector.serpapi_key != "yout_serp_api_key":
        serpapi_status_msg, serpapi_status_class = "✅ Active (External API)", "active"
    elif using_default_serpapi_key:
        serpapi_status_msg, serpapi_status_class = "⚠️ Active (Using Default Key)", "default-key"
    else:
        serpapi_status_msg, serpapi_status_class = "❌ Not Set / Limited Functionality", "not-set"
    st.markdown(f"<div class='serpapi-status {serpapi_status_class}'>**{serpapi_status_msg}**</div>", unsafe_allow_html=True)
    st.info("Valid SerpAPI key ensures full web-based plagiarism checks.")

# --- Input Section ---
st.header("📝 Document Input")
input_tab1, input_tab2 = st.tabs(["Upload .txt File", "Enter Text Manually"])
student_text = ""
document_name = "Untitled Document"

with input_tab1:
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt", key="student_file_uploader")
    if uploaded_file:
        student_text = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        document_name = uploaded_file.name
        st.success(f"File '{document_name}' loaded successfully! ✅")
        st.expander("Preview Document").text_area("File content:", student_text, height=150, disabled=True)

with input_tab2:
    manual_text = st.text_area("Paste your text here", height=300, placeholder="Enter text...")
    if manual_text:
        student_text = manual_text
        document_name = "Manual Input"
        st.success("Text entered successfully! ✅")

# --- Internal Documents (Optional) ---
st.header("📚 Internal Comparison (Optional)")
known_docs_files = st.file_uploader("Choose multiple .txt files", type="txt", accept_multiple_files=True)
known_documents_list = [f.getvalue().decode("utf-8") for f in known_docs_files] if known_docs_files else []

# --- Run Analysis ---
st.header("🚀 Run Analysis")
if st.button("Analyze Document for Plagiarism", type="primary", use_container_width=True):
    if not student_text:
        st.error("Please provide text before running the analysis.")
    else:
        with st.spinner("Analyzing document..."):
            try:
                analysis_result = detector.compare_documents(
                    student_text,
                    threshold=threshold,
                    num_rephrase_suggestions=num_rephrase_suggestions,
                    known_documents=known_documents_list
                )
                st.success("Analysis Complete! 🎉")
                st.markdown("---")
                st.header(f"📊 Plagiarism Report for **'{document_name}'**")
                report_col1, report_col2 = st.columns([1, 1.2])

                # --- Column 1: Details ---
                with report_col1:
                    overall = analysis_result.get("overall_similarity", 0.0)
                    plagiarized_count = sum(1 for s in analysis_result["sentences"] if s["is_plagiarized"])
                    total_sentences = len(analysis_result["sentences"])
                    st.markdown(f"**Overall Plagiarism Score:** <span style='font-size:1.8rem;color:#e67e22;font-weight:bold;'>{overall:.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Plagiarized Sentences:** <span style='font-size:1.8rem;color:#c0392b;font-weight:bold;'>{plagiarized_count}</span> / {total_sentences}", unsafe_allow_html=True)
                    st.markdown("### Sentence-by-Sentence Breakdown")
                    if plagiarized_count == 0:
                        st.info("🥳 No highly similar sentences found.")
                    else:
                        for i, r in enumerate(analysis_result["sentences"], 1):
                            if r['is_plagiarized']:
                                with st.expander(f"🚨 Plagiarized Sentence {i} (Similarity: {r['similarity']:.2f})"):
                                    st.markdown(f"**Original Text:** <mark class='plagiarized'>{r['sentence']}</mark>", unsafe_allow_html=True)
                                    st.markdown(f"**Similarity Score:** <span style='color:#d32f2f; font-weight:bold;'>{r['similarity']:.2f}</span>", unsafe_allow_html=True)
                                    if r['source_url']: st.markdown(f"**Potential Web Source:** [Link]({r['source_url']})")
                                    if r['source_text_match']: st.markdown(f"**Matching Snippet:** *\"...{r['source_text_match'][:350]}...\"*")
                                    st.markdown("**💡 Rephrasing Suggestions:**")
                                    if r['rephrases']:
                                        for suggestion in r['rephrases']:
                                            st.write(f"- {suggestion}")
                                    else:
                                        st.write("*(No rephrasing suggestions generated.)*")
                                st.markdown("---")

                # --- Column 2: Document View ---
                with report_col2:
                    st.subheader("Document View with Highlights")
                    original_sentences_for_display = detector._split_sentences(student_text)
                    highlighted_output_parts = []
                    analysis_sentences_map = {s['sentence']: s for s in analysis_result['sentences']}

                    for sent in original_sentences_for_display:
                        stripped_sent = sent.strip()
                        entry = analysis_sentences_map.get(stripped_sent)
                        if entry and entry["is_plagiarized"]:
                            highlighted_output_parts.append(f"<mark class='plagiarized' title='Similarity: {entry['similarity']:.2f}'>{sent}</mark>")
                        else:
                            highlighted_output_parts.append(sent)

                    highlighted_document_html = " ".join(highlighted_output_parts).replace("\n\n", "<br><br>").replace("\n", " ").strip()
                    st.markdown(f"<div style='border:1px solid #dcdcdc; padding:20px; border-radius:8px; background-color:white; overflow-y:auto; max-height:700px; line-height:1.6;'>{highlighted_document_html}</div>", unsafe_allow_html=True)

                    st.markdown("""
                    <p style='font-size: small; margin-top: 20px; text-align: center;'>
                        <strong>Highlight Legend:</strong>
                        <span style='background-color:#fff700; color:#c70000; padding: 4px 8px; border-radius:5px; font-weight:bold; border:1px solid #c70000;' title='Plagiarized sentence'>Potentially Plagiarized Sentence</span>
                        <br/>
                        <span style='padding: 4px 8px; border-radius:5px;' title='Original / Non-Plagiarized sentence'>Normal Text</span>
                    </p>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit, Sentence Transformers, and SerpAPI.")
