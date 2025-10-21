# ============================================================
# 🔍 Semantic Plagiarism Detector + Web Search (SerpAPI)
#      Cloud-safe version for Streamlit deployment
# ============================================================

import os
import logging
import numpy as np
import torch
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# Attempt to import transformers; handle missing backends
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        # SentenceTransformer for semantic similarity
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Loaded SentenceTransformer: {model_name}")

        # Detect if running on Streamlit Cloud
        running_on_streamlit_cloud = os.environ.get("STREAMLIT_SERVER_RUNNING") == "true"

        # Use lightweight T5 model on Streamlit Cloud to avoid tokenizer issues
        if running_on_streamlit_cloud:
            logger.info("Running on Streamlit Cloud — using lightweight t5-small model")
            rephrase_model_name = "t5-small"

        # T5 paraphrasing model
        self.rephrase_model = None
        self.rephrase_tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                set_seed(seed)
                self.rephrase_tokenizer = AutoTokenizer.from_pretrained(rephrase_model_name, use_fast=False)
                self.rephrase_model = AutoModelForSeq2SeqLM.from_pretrained(rephrase_model_name)
                if self.device == "cuda":
                    self.rephrase_model.to("cuda")
                self.rephrase_model.eval()
                logger.info(f"Loaded paraphrase model: {rephrase_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load '{rephrase_model_name}' model. Paraphrasing disabled. Error: {e}")

        # SerpAPI key
        self.serpapi_key = serpapi_key if SERPAPI_AVAILABLE else None
        if not self.serpapi_key:
            logger.warning("SerpAPI not available or key missing. Web plagiarism search will be limited.")

        self._rephrase_cache = {}

    # -------------------------
    # Document Utilities
    # -------------------------
    def load_document_text(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded document: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""

    # -------------------------
    # Sentence Utilities
    # -------------------------
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+|\n', text)
        return [s.strip() for s in sentences if s.strip()]

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
    # T5 Paraphrasing
    # -------------------------
    def _t5_generate_paraphrases(self, sentence: str, num_suggestions: int = 3):
        if not self.rephrase_model:
            return ["Paraphrasing not available (missing backend)"]

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
                num_return_sequences=max(6, num_suggestions*2),
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.5
            )

            raw = [self.rephrase_tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
            unique = []
            for r in raw:
                if r and r.lower() not in [u.lower() for u in unique] and len(r.split()) >= 5 and r.lower() != sentence.lower():
                    unique.append(r)
            suggestions = unique[:num_suggestions]
            if not suggestions:
                suggestions = ["No effective rephrasing suggestions could be generated."]
            self._rephrase_cache[cache_key] = suggestions
            return suggestions
        except Exception as e:
            logger.error(f"T5 paraphrasing failed: {e}")
            return ["No effective rephrasing suggestions could be generated."]

    # -------------------------
    # SerpAPI Web Search
    # -------------------------
    def search_web_source(self, query: str) -> Dict[str, str]:
        if not self.serpapi_key:
            return {"link": None, "snippet": None}
        try:
            search = GoogleSearch({"engine": "google", "q": query, "api_key": self.serpapi_key, "num": "1"})
            results = search.get_dict()
            if "organic_results" in results and results["organic_results"]:
                first_result = results["organic_results"][0]
                return {"link": first_result.get("link"), "snippet": first_result.get("snippet")}
        except Exception as e:
            logger.warning(f"Web search failed for '{query}': {e}")
        return {"link": None, "snippet": None}

    # -------------------------
    # Semantic Similarity
    # -------------------------
    def _get_semantic_similarity(self, sentence1: str, sentence2: str) -> float:
        if not sentence1 or not sentence2:
            return 0.0
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    # -------------------------
    # Compare Documents
    # -------------------------
    def compare_documents(
        self,
        student_text: str,
        threshold: float = 0.7,
        num_rephrase_suggestions: int = 3,
        known_documents: List[str] = None
    ):
        student_emb, student_sentences = self._get_sentence_embeddings(student_text)
        if student_emb is None:
            return {"overall_similarity": 0.0, "sentences": []}

        report = []
        similarity_scores = []

        # Process known documents
        known_doc_embeddings = []
        known_doc_sentences = []
        if known_documents:
            for doc in known_documents:
                doc_emb, doc_sents = self._get_sentence_embeddings(doc)
                if doc_emb is not None:
                    known_doc_embeddings.append(doc_emb)
                    known_doc_sentences.extend(doc_sents)

        for student_sentence in student_sentences:
            current_max_sim = 0.0
            found_source_url = None
            found_source_text = None

            # Check internal docs
            if known_doc_embeddings:
                for known_sent in known_doc_sentences:
                    sim = self._get_semantic_similarity(student_sentence, known_sent)
                    if sim > current_max_sim:
                        current_max_sim = sim
                        found_source_text = f"Internal doc: '{known_sent[:50]}...'"

            # Check web search
            web_result = self.search_web_source(student_sentence)
            if web_result["link"] and web_result["snippet"]:
                web_snip_sim = self._get_semantic_similarity(student_sentence, web_result["snippet"])
                if web_snip_sim > current_max_sim:
                    current_max_sim = web_snip_sim
                    found_source_url = web_result["link"]
                    found_source_text = web_result["snippet"]

            is_plagiarized = current_max_sim >= threshold
            rephrases = self._t5_generate_paraphrases(student_sentence, num_rephrase_suggestions) if is_plagiarized else []

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
