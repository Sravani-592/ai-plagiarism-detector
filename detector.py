# ============================================================
# 🔍 Semantic Plagiarism Detector + Web Search (SerpAPI)
#      with User Document Upload and Highlighted Output
# ============================================================

# Install required packages if not already installed:
# !pip install sentence-transformers transformers torch scikit-learn serpapi
# !pip install google-search-results requests beautifulsoup4

import os
import logging
import numpy as np
import torch
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ANSI escape codes for coloring console output (though not used in Streamlit UI directly, still good to keep for console logging)
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BACKGROUND_RED = '\033[41m'
    BACKGROUND_YELLOW = '\033[43m' # Using yellow background for highlights

# -------------------------
# Main Class
# -------------------------
class SemanticPlagiarismDetector:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        rephrase_model_name: str = "Vamsi/T5_Paraphrase_Paws",
        seed: int = 42,
        serpapi_key: str = None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # SentenceTransformer for similarity
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Loaded SentenceTransformer: {model_name}")

        # T5 paraphrasing model
        set_seed(seed)
        self.rephrase_tokenizer = AutoTokenizer.from_pretrained(rephrase_model_name)
        self.rephrase_model = AutoModelForSeq2SeqLM.from_pretrained(rephrase_model_name)
        if self.device == "cuda":
            self.rephrase_model.to("cuda")
        self.rephrase_model.eval()
        logger.info(f"Loaded paraphrase model: {rephrase_model_name}")

        # SerpAPI key
        self.serpapi_key = serpapi_key
        if not serpapi_key:
            logger.warning("No SerpAPI key provided. Web plagiarism source will not be detected.")

        self._rephrase_cache = {}

    # -------------------------
    # Document Loading Utility
    # -------------------------
    def load_document_text(self, file_path: str) -> str:
        """
        Loads text content from a .txt file.
        For other formats (docx, pdf), additional libraries would be needed.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully loaded document from {file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"Error: File not found at {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""

    # -------------------------
    # Sentence utilities
    # -------------------------
    def _split_sentences(self, text: str) -> List[str]:
        # Improved sentence splitting
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+|\n', text)
        return [s.strip() for s in sentences if s.strip()]

    def _normalize(self, s: str) -> str:
        return s.strip()

    # -------------------------
    # Embeddings
    # -------------------------
    def _get_sentence_embeddings(self, text: str):
        sentences = self._split_sentences(text)
        if not sentences:
            return None, []
        embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        return embeddings, sentences

    # -------------------------
    # T5 paraphrasing
    # -------------------------
    def _t5_generate_paraphrases(self, sentence: str, num_suggestions: int = 3):
        cache_key = (sentence, num_suggestions)
        if cache_key in self._rephrase_cache:
            return self._rephrase_cache[cache_key]

        prompt = f"paraphrase: {sentence}"
        try:
            input_ids = self.rephrase_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            if self.device == "cuda":
                input_ids = input_ids.to("cuda")

            outputs = self.rephrase_model.generate(
                input_ids,
                max_length=64,
                min_length=10,
                num_return_sequences=max(6, num_suggestions*2), # Generate more and filter for uniqueness
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.5
            )

            raw = [self.rephrase_tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
            unique = []
            # Filter for uniqueness and minimum length
            for r in raw:
                if r and r.lower() not in [u.lower() for u in unique] and len(r.split()) >= 5 and r.lower() != sentence.lower():
                    unique.append(r)
            suggestions = unique[:num_suggestions]
            if not suggestions: # Fallback if no good unique suggestions are found
                 suggestions = ["No effective rephrasing suggestions could be generated that are sufficiently different."]
            self._rephrase_cache[cache_key] = suggestions
            return suggestions
        except Exception as e:
            logger.error(f"T5 paraphrasing failed: {e}")
            return ["No effective rephrasing suggestions could be generated."]

    # -------------------------
    # SerpAPI Web Search & Content Fetching
    # -------------------------
    def search_web_source(self, query: str) -> Dict[str, str]:
        """
        Searches the web for the query and returns the top result's link and snippet.
        """
        if not self.serpapi_key:
            return {"link": None, "snippet": None}
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "num": "1" # Only interested in the top result for comparison
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            if "organic_results" in results and results["organic_results"]:
                first_result = results["organic_results"][0]
                return {
                    "link": first_result.get("link"),
                    "snippet": first_result.get("snippet")
                }
        except Exception as e:
            logger.warning(f"Web search failed for query '{query}': {e}")
        return {"link": None, "snippet": None}

    def _get_semantic_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculates semantic similarity between two sentences."""
        if not sentence1 or not sentence2:
            return 0.0
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        # Cosine similarity returns a tensor, take the first element (similarity between sentence1 and sentence2)
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    # -------------------------
    # Compare Documents (Enhanced for Plagiarism Detection)
    # -------------------------
    def compare_documents(
        self,
        student_text: str,
        threshold: float = 0.7,
        num_rephrase_suggestions: int = 3,
        known_documents: List[str] = None # Optional: a list of known texts to check against
    ):
        student_emb, student_sentences = self._get_sentence_embeddings(student_text)
        if student_emb is None:
            return {"overall_similarity": 0.0, "sentences": []}

        report = []
        similarity_scores = []

        # Process known documents if provided
        known_doc_embeddings = []
        known_doc_sentences = []
        if known_documents:
            for doc in known_documents:
                doc_emb, doc_sents = self._get_sentence_embeddings(doc)
                if doc_emb is not None:
                    known_doc_embeddings.append(doc_emb)
                    known_doc_sentences.extend(doc_sents)

        for i, student_sentence in enumerate(student_sentences):
            current_max_sim = 0.0
            found_source_url = None
            found_source_text = None

            # 1. Check against known internal documents (if any)
            if known_doc_embeddings:
                for known_sent_idx, known_sent in enumerate(known_doc_sentences):
                    sim = self._get_semantic_similarity(student_sentence, known_sent)
                    if sim > current_max_sim:
                        current_max_sim = sim
                        found_source_text = f"Internal document: '{known_sent[:50]}...'" # Truncate for display

            # 2. Check against web search results
            web_search_result = self.search_web_source(student_sentence)
            if web_search_result["link"] and web_search_result["snippet"]:
                web_snippet_sim = self._get_semantic_similarity(student_sentence, web_search_result["snippet"])
                if web_snippet_sim > current_max_sim:
                    current_max_sim = web_snippet_sim
                    found_source_url = web_search_result["link"]
                    found_source_text = web_search_result["snippet"]

            # Determine if plagiarized based on threshold
            is_plagiarized = current_max_sim >= threshold

            rephrases = []
            if is_plagiarized: # Only generate rephrases if deemed plagiarized
                rephrases = self._t5_generate_paraphrases(student_sentence, num_rephrase_suggestions)

            report.append({
                "sentence": student_sentence,
                "similarity": current_max_sim,
                "is_plagiarized": is_plagiarized,
                "source_url": found_source_url,
                "source_text_match": found_source_text,
                "rephrases": rephrases
            })
            similarity_scores.append(current_max_sim)

        overall_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        return {"overall_similarity": overall_similarity, "sentences": report}

    # The print_report and analyze_document_and_highlight methods from the original script
    # are no longer directly used in the Streamlit app's display logic, but you can keep
    # them in detector.py if you have other uses for them (e.g., console-based testing).
    # For Streamlit, the display logic is directly in app.py.
