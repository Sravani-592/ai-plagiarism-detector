# AI Semantic Plagiarism Detector

This project implements an AI-powered semantic plagiarism detector using Streamlit for the frontend, Sentence Transformers for semantic similarity, and SerpAPI for web-based source checking and rephrasing suggestions.

## Features

*   **Text Input**: Upload `.txt` files or paste text directly.
*   **Semantic Analysis**: Detects plagiarism based on the meaning of sentences, not just exact word matches.
*   **Web Source Matching**: Utilizes SerpAPI to find potential online sources for highly similar sentences.
*   **Internal Document Comparison**: Allows uploading multiple `.txt` files to check against private known documents.
*   **Rephrasing Suggestions**: Generates alternative phrasings for plagiarized sentences.
*   **Interactive Report**: Provides a sentence-by-sentence breakdown and a highlighted document view.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sravani-592/ai-plagiarism-detector.git
    cd ai-plagiarism-detector
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK 'punkt' tokenizer data:**
    You need to download the `punkt` tokenizer for NLTK. You can do this by running a Python interpreter once:
    ```python
    import nltk
    nltk.download('punkt')
    ```
    Or, you can include the robust download logic directly in your `detector.py` or `app.py` for deployment environments:
    ```python
    # In detector.py or app.py (ideally detector.py at the top)
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    ```

5.  **Set up SerpAPI Key:**
    The application uses SerpAPI for web searches.
    *   **Get your API Key:** Sign up at [SerpAPI](https://serpapi.com/) to get your API key.
    *   **Securely set the key:**
        *   **For local development:** In .streamlit folder there is secrets file inside that add your serp api key a:
            ```
            SERPAPI_KEY="YOUR_SERPAPI_KEY_HERE"
            ```
            and in app.py place your Serp_api_key at line number 104, and 150.
## How to Run

1.  Ensure you have followed the setup steps, including setting your SerpAPI key.
2.  Run the Streamlit app from your project root:
    ```bash
    streamlit run app.py
    ```

## Project Structure

ai-plagiarism-detector


├── app.py # Main Streamlit application


├── detector.py # Core plagiarism detection logic (SemanticPlagiarismDetector class)


├── requirements.txt # Python dependencies


├── .gitignore # Files/folders to ignore by Git


└── README.md # Project description and instructions
