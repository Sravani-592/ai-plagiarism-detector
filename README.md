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
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
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
        *   **For local development:** Create a file `.env` in your project root (add `*.env` to `.gitignore`) and add:
            ```
            SERPAPI_KEY="YOUR_SERPAPI_KEY_HERE"
            ```
            Then, in your Streamlit app, `os.environ.get("SERPAPI_KEY")` will pick this up.
        *   **For Streamlit Cloud:** Use the "Secrets" management interface in your app's settings. Add a secret named `SERPAPI_KEY` with your API key as its value.
        *   **Important:** **DO NOT** commit your SerpAPI key directly into any code file or `.streamlit/secrets.toml` to a public repository. The `.gitignore` file is set up to prevent `secrets.toml` from being committed.

## How to Run

1.  Ensure you have followed the setup steps, including setting your SerpAPI key.
2.  Run the Streamlit app from your project root:
    ```bash
    streamlit run app.py
    ```

## Project Structure