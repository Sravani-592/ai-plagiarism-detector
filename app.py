import streamlit as st
import os
import io

# --- Import SemanticPlagiarismDetector class from detector.py ---
# Assuming detector.py is in the same directory or accessible via PYTHONPATH
from detector import SemanticPlagiarismDetector

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Semantic Plagiarism Detector",
    initial_sidebar_state="expanded" # Start with sidebar open
)

# --- Custom CSS for a cleaner look ---
st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: #f0f2f6; /* Light gray background for a professional feel */
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem; /* Slightly larger tab text */
    }
    h1 {
        color: #1a1a2e; /* Darker blue for main title */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h2, h3 {
        color: #2e2e4f; /* Slightly lighter than h1 */
    }
    .stButton > button {
        background-color: #4CAF50; /* Green button for primary action */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    /* Style for the text area */
    .stTextArea > label {
        font-weight: bold;
        color: #2e2e4f;
    }
    .stFileUploader > label {
        font-weight: bold;
        color: #2e2e4f;
    }
    /* Custom style for highlight - NOW YELLOW */
    mark.plagiarized {
        background-color: #fff700; /* Bright yellow background */
        color: #c70000; /* Dark red text for contrast */
        padding: 2px 3px;
        border-radius: 3px;
        font-weight: bold;
    }
    /* Info box for SerpAPI status */
    .serpapi-status {
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-weight: bold;
        color: #333;
    }
    .serpapi-status.active {
        background-color: #e8f5e9; /* Light green */
        border: 1px solid #4CAF50;
    }
    .serpapi-status.default-key {
        background-color: #fffde7; /* Light yellow */
        border: 1px solid #ffeb3b;
    }
    .serpapi-status.not-set {
        background-color: #ffebee; /* Light red */
        border: 1px solid #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Instantiate the detector globally or in a cached function ---
@st.cache_resource
def get_plagiarism_detector_instance():
    serpapi_key = None

    # 1. Try to get from st.secrets (recommended for Streamlit Cloud/production)
    if hasattr(st, 'secrets'):
        serpapi_key = st.secrets.get("SERPAPI_KEY")

    # 2. If not found in secrets, try from environment variable (good for local dev)
    if not serpapi_key:
        serpapi_key = os.environ.get("SERPAPI_KEY")

    # 3. Fallback to a placeholder/default key if still not found, but warn the user
   
    DEFAULT_SERPAPI_KEY = "Your_Serp_Api_Key"
    if not serpapi_key:
        serpapi_key = DEFAULT_SERPAPI_KEY

    detector_instance = SemanticPlagiarismDetector(serpapi_key=serpapi_key)

    # Provide a warning if the key is the hardcoded default and not from a secure source
    if serpapi_key == DEFAULT_SERPAPI_KEY and \
       (not (hasattr(st, 'secrets') and st.secrets.get("SERPAPI_KEY") == serpapi_key) and \
        not (os.environ.get("SERPAPI_KEY") == serpapi_key)):
        st.warning(
            "⚠️ **SerpAPI key is being used from a default/hardcoded value.** "
            "For full web search functionality and to avoid potential rate limits, "
            "please set your SerpAPI key securely in `.streamlit/secrets.toml` "
            "or as an environment variable (`SERPAPI_KEY`)."
        )
    return detector_instance, serpapi_key == DEFAULT_SERPAPI_KEY

detector, using_default_serpapi_key = get_plagiarism_detector_instance()

# --- Header ---
st.title("📚 AI Semantic Plagiarism Detector")
st.markdown("A tool to analyze text for semantic similarity against web sources and internal documents.")

st.markdown("---")

# --- Sidebar for Settings (Optional, but can declutter main page) ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Adjust the analysis parameters here.")

    threshold = st.slider(
        "Plagiarism Threshold (Similarity Score)",
        min_value=0.60, max_value=0.99, value=0.80, step=0.01,
        help="Similarity score above which a sentence is considered potentially plagiarized."
    )
    num_rephrase_suggestions = st.slider(
        "Number of Rephrasing Suggestions",
        min_value=1, max_value=5, value=3,
        help="How many alternative phrasings to generate for detected sentences. Requires an active SerpAPI key."
    )

    st.markdown("---")
    st.subheader("SerpAPI Status")
    serpapi_status_msg = ""
    serpapi_status_class = ""
    if detector.serpapi_key and detector.serpapi_key != "Your_Serp_Api_key":
        serpapi_status_msg = "✅ Active (External API)"
        serpapi_status_class = "active"
    elif using_default_serpapi_key:
        serpapi_status_msg = "⚠️ Active (Using Default Key)"
        serpapi_status_class = "default-key"
    else:
        serpapi_status_msg = "❌ Not Set / Limited Functionality"
        serpapi_status_class = "not-set"
    st.markdown(f"<div class='serpapi-status {serpapi_status_class}'>**{serpapi_status_msg}**</div>", unsafe_allow_html=True)
    st.info("A valid SerpAPI key is crucial for comprehensive web-based plagiarism checks and rephrasing suggestions.")

# --- Main Content Area ---

# --- Input Section ---
st.header("📝 Document Input")
input_tab1, input_tab2 = st.tabs(["Upload .txt File", "Enter Text Manually"])

student_text = ""
document_name = "Untitled Document"

with input_tab1:
    uploaded_file = st.file_uploader(
        "Choose a .txt file to analyze",
        type="txt",
        key="student_file_uploader",
        help="Upload your document here. Only plain text (.txt) files are supported."
    )
    if uploaded_file is not None:
        string_io = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        student_text = string_io.read()
        document_name = uploaded_file.name
        st.success(f"File '{document_name}' loaded successfully! ✅")
        st.expander("Preview Document").text_area("File content:", student_text, height=150, disabled=True)
    else:
        st.info("Please upload a .txt file to begin analysis.")

with input_tab2:
    manual_text = st.text_area(
        "Paste your text here",
        height=300,
        placeholder="Enter the document text you want to analyze for plagiarism...",
        help="You can paste your essay, article, or any text directly into this box."
    )
    if manual_text:
        student_text = manual_text
        document_name = "Manual Input"
        st.success("Text entered successfully! ✅")
    else:
        st.info("Paste your content into the text area to proceed.")

st.markdown("---")

# --- Known Documents (Optional) ---
st.header("📚 Internal Comparison (Optional)")
st.info("Upload documents here if you want to check for plagiarism against your own internal sources (e.g., other student essays, specific articles, past submissions).")
known_docs_files = st.file_uploader(
    "Choose multiple .txt files for internal comparison",
    type="txt",
    accept_multiple_files=True,
    key="known_docs_uploader",
    help="These documents will be compared against the main input document. Only .txt files."
)
known_documents_list = []
if known_docs_files:
    for doc_file in known_docs_files:
        known_documents_list.append(doc_file.getvalue().decode("utf-8"))
    st.success(f"Loaded {len(known_documents_list)} known document(s) for internal comparison. ✅")
else:
    st.info("No internal documents uploaded. Analysis will rely primarily on web search.")

st.markdown("---")

# --- Analysis Button ---
st.header("🚀 Run Analysis")
if st.button("Analyze Document for Plagiarism", type="primary", use_container_width=True):
    if not student_text:
        st.error("Please provide text either by uploading a file or entering it manually before running the analysis.")
    else:
        with st.spinner("Analyzing document... This might take a moment, especially with web searches and paraphrase generation. Please be patient."):
            try:
                analysis_result = detector.compare_documents(
                    student_text,
                    threshold=threshold,
                    num_rephrase_suggestions=num_rephrase_suggestions,
                    known_documents=known_documents_list
                )
                st.success("Analysis Complete! 🎉")
                # Removed st.balloons() as requested

                st.markdown("---")
                st.header(f"📊 Plagiarism Report for **'{document_name}'**")

                # --- Two-Pane Layout for Report ---
                report_col1, report_col2 = st.columns([1, 1.2]) # Adjust column width ratio

                with report_col1:
                    st.subheader("Detailed Plagiarism Insights")
                    overall = analysis_result.get("overall_similarity", 0.0)
                    plagiarized_count = sum(1 for s in analysis_result["sentences"] if s["is_plagiarized"])
                    total_sentences = len(analysis_result["sentences"])

                    st.markdown(f"**Overall Document Plagiarism Score:** <span style='font-size: 1.8rem; color: #e67e22; font-weight: bold;'>{overall:.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Plagiarized Sentences Detected:** <span style='font-size: 1.8rem; color: #c0392b; font-weight: bold;'>{plagiarized_count}</span> / {total_sentences}", unsafe_allow_html=True)

                    st.markdown("### Sentence-by-Sentence Breakdown")
                    if plagiarized_count == 0:
                        st.info("🥳 **Congratulations! No highly similar sentences found.** Your document appears original based on the current threshold and sources checked.")
                    else:
                        for i, r in enumerate(analysis_result["sentences"], 1):
                            if r['is_plagiarized']:
                                with st.expander(f"🚨 Plagiarized Sentence {i} (Similarity: {r['similarity']:.2f})"):
                                    st.markdown(f"**Original Text:** <mark class='plagiarized'>{r['sentence']}</mark>", unsafe_allow_html=True)
                                    st.markdown(f"**Similarity Score:** <span style='color:#d32f2f; font-weight:bold;'>{r['similarity']:.2f}</span> (Threshold: {threshold:.2f})", unsafe_allow_html=True)

                                    if r['source_url']:
                                        st.markdown(f"**Potential Web Source:** [Link]({r['source_url']})")
                                    if r['source_text_match']:
                                        st.markdown(f"**Matching Snippet:** *\"...{r['source_text_match'][:350]}...\"*")
                                    else:
                                        st.markdown("*(No specific web snippet found, but internal document similarity was high or a broader web match was detected.)*")

                                    st.markdown("**💡 Rephrasing Suggestions:**")
                                    if r['rephrases']:
                                        for suggestion in r['rephrases']:
                                            st.write(f"- {suggestion}")
                                    else:
                                        st.write("*(No effective rephrasing suggestions could be generated for this sentence, or SerpAPI key is inactive.)*")
                                st.markdown("---") # Separator between plagiarized sentences

                with report_col2:
                    st.subheader("Document View with Highlights")
                    highlighted_output_parts = []
                    # Use the same sentence splitting logic as the detector for consistency
                    original_sentences_for_display = detector._split_sentences(student_text)

                    # Create a mapping from original sentences to analysis results
                    # This handles potential discrepancies if splitting logic differs slightly
                    analysis_sentence_map = {entry['sentence']: entry for entry in analysis_result["sentences"]}

                    for i, original_sentence in enumerate(original_sentences_for_display):
                        # Find the corresponding analysis result for this sentence
                        # Use stripping to match as _split_sentences also strips
                        stripped_original = original_sentence.strip()
                        matched_analysis = next((entry for entry in analysis_result["sentences"] if entry['sentence'].strip() == stripped_original), None)

                        if matched_analysis and matched_analysis["is_plagiarized"]:
                            # Use the custom CSS class for highlighting
                            highlighted_output_parts.append(f"<mark class='plagiarized'>{original_sentence}</mark>")
                        else:
                            highlighted_output_parts.append(original_sentence)

                    # Join parts, attempting to preserve paragraph structure.
                    # Replace single newlines with a space for flow, but double newlines for paragraphs
                    highlighted_document_html = " ".join(highlighted_output_parts).replace("\n\n", "<br><br>").replace("\n", " ").strip()

                    st.markdown(
                        f"<div style='border: 1px solid #dcdcdc; padding: 20px; border-radius: 8px; background-color: white; overflow-y: auto; max-height: 700px; line-height: 1.6;'>{highlighted_document_html}</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown("""
                    <p style='font-size: small; margin-top: 20px; text-align: center;'>
                        <strong>Highlight Legend:</strong>
                        <span style='background-color:#fff700; color:#c70000; padding: 4px 8px; border-radius: 5px; font-weight: bold; border: 1px solid #c70000;'>
                            Potentially Plagiarized Sentence
                        </span>
                        <br/>
                        <span style='padding: 4px 8px; border-radius: 5px;'>Normal Text: Original/Non-Plagiarized Sentence</span>
                    </p>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.error("Please check your input text and API key configuration.")
    # else: # This else is now handled by the check inside the button click
    #     st.warning("Please upload a document or enter text to analyze before running the analysis.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit, Sentence Transformers, and SerpAPI.")
