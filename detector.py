# ============================================================
# 🔍 Semantic Plagiarism Detector + Web Search (SerpAPI)
#      Improved version for Streamlit deployment
# ============================================================

import os
import logging
import numpy as np
import torch
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from serpapi import GoogleSearch

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

        # Load SentenceTransformer
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Loaded SentenceTransformer: {model_name}")

        # Detect Streamlit Cloud
        running_on_streamlit_cloud = os.environ.get("STREAMLIT_SERVER_RUNNING") == "true"
        if running_on_streamlit_cloud:
            logger.info("Running on Streamlit Cloud — using lightweight t5-small model")
            rephrase_model_name = "t5-small"

        # Load T5 paraphrasing model safely
        set_seed(seed)
        self.rephrase_model = None
        self.rephrase_tokenizer = None
        try:
            self.rephrase_tokenizer = AutoTokenizer.from_pretrained(rephrase_model_name, use_fast=False)
            self.rephrase_model = AutoModelForSeq2SeqLM.from_pretrained(rephrase_model_name)
            if self.device == "cuda":
                self.rephrase_model.to("cuda")
            self.rephrase_model.eval()
            logger.info(f"Loaded paraphrase model: {rephrase_model_name}")
        except Exception as e:
            logger.warning(f"Paraphrasing model failed to load: {e}")
            self.rephrase_model = None
            self.rephrase_tokenizer = None

        # SerpAPI key
        self.serpapi_key = serpapi_key
        if not serpapi_key:
            logger.warning("No SerpAPI key provided. Web plagiarism search will be limited.")

        self._rephrase_cache = {}

    # -------------------------
    # Text normalization
    # -------------------------
    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    # -------------------------
    # Sentence utilities
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
        sentences_norm = [self._normalize_text(s) for s in sentences]
        embeddings = self.model.encode(sentences_norm, convert_to_tensor=True, show_progress_bar=False)
        return embeddings, sentences

    # -------------------------
    # T5 Paraphrasing
    # -------------------------
    def _t5_generate_paraphrases(self, sentence: str, num_suggestions: int = 3):
        if not self.rephrase_model or not self.rephrase_tokenizer:
            return ["Paraphrasing unavailable due to missing backend."]
        
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
        params = {"engine": "google", "q": query, "api_key": self.serpapi_key, "num": "1"}
        try:
            search = GoogleSearch(params)
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
        sentence1_norm = self._normalize_text(sentence1)
        sentence2_norm = self._normalize_text(sentence2)
        if not sentence1_norm or not sentence2_norm:
            return 0.0
        embeddings = self.model.encode([sentence1_norm, sentence2_norm], convert_to_tensor=True)
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

        # Prepare known document sentences
        known_doc_sentences = []
        if known_documents:
            for doc in known_documents:
                _, doc_sents = self._get_sentence_embeddings(doc)
                known_doc_sentences.extend(doc_sents)

        for student_sentence in student_sentences:
            current_max_sim = 0.0
            found_source_url = None
            found_source_text = None

            # Internal documents
            for known_sent in known_doc_sentences:
                sim = self._get_semantic_similarity(student_sentence, known_sent)
                if sim > current_max_sim:
                    current_max_sim = sim
                    found_source_text = f"Internal doc: '{known_sent[:100]}...'"

            # Web snippet
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
