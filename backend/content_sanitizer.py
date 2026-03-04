import re
from collections import Counter
from typing import List, Set

try:
    import spacy
    # Load basic language model, disable heavy pipelines for speed
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except (ImportError, Exception):
    nlp = None

class ContentSanitizer:
    def __init__(self):
        # Baseline noise patterns for non-conceptual framing (regex)
        self.framing_patterns = [
            r"(?i)\b(subscribe( to my channel)?|like and subscribe|hit the bell icon)\b",
            r"(?i)\b(thank you for watching|thanks for watching|thanks for joining)\b",
            r"(?i)\b(welcome to this( lecture| video| session)?)\b",
            r"(?i)\b(in this( lecture| video| session) we will( discuss)?)\b",
            r"(?i)\b(slide \d+|chapter \d+|unit \d+)\b",
            r"(?i)\b(important questions? for (sppu|exam)|exam oriented)\b",
        ]
        self.repeated_ocr_noise = set()

    def build_dynamic_ocr_noise(self, ocr_texts: List[str], min_occurrence_ratio: float = 0.3):
        """
        Analyzes all OCR slide texts to find repeating headers/footers (presentation metadata).
        """
        if not ocr_texts:
            return
            
        total_slides = len(ocr_texts)
        line_counts = Counter()
        
        for text in ocr_texts:
            if not text:
                continue
            # Extract distinct lines (headers usually occupy their own line)
            lines = set([line.strip() for line in text.split('\n') if len(line.strip()) > 4])
            for line in lines:
                line_counts[line] += 1
                
        # Register lines that appear on more than X% of slides
        threshold = max(2, int(total_slides * min_occurrence_ratio))
        for line, count in line_counts.items():
            if count >= threshold:
                self.repeated_ocr_noise.add(line)
        
        if self.repeated_ocr_noise:
            print(f"[Sanitizer] Identified {len(self.repeated_ocr_noise)} dynamic OCR noise phrases (e.g. headers).")

    def is_concept_dense(self, sentence: str) -> bool:
        """
        Check if a sentence has actual educational conceptual value.
        """
        words = sentence.split()
        if len(words) < 4:
            return False # Too short to be a meaningful concept
            
        # Structural check: too much punctuation often means OCR artifact/code snippet
        punct_count = sum(1 for c in sentence if c in r"|/\{}[]()<>*^%$#@!+=" "")
        if len(sentence) > 0 and (punct_count / len(sentence)) > 0.15:
            return False
            
        if nlp:
            doc = nlp(sentence)
            # A valid educational sentence should have a noun/propn and a verb
            has_noun = any(token.pos_ in ["NOUN", "PROPN"] for token in doc)
            has_verb = any(token.pos_ == "VERB" for token in doc)
            if not (has_noun and has_verb):
                return False
            
        return True

    def sanitize_sentence(self, sentence: str) -> str:
        s = sentence.strip()
        
        # 1. OCR specific dynamic noise filter (watermarks, slide headers)
        for noise_line in self.repeated_ocr_noise:
            if noise_line in s:
                s = s.replace(noise_line, "")
                
        # 2. Heuristic regex filter for standard tutorial noise
        for pattern in self.framing_patterns:
            s = re.sub(pattern, "", s)
            
        return s.strip()

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize a full paragraph/document of text.
        """
        if not text:
            return ""
            
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except ImportError:
            sentences = text.split(". ")
            
        sanitized = []
        for sent in sentences:
            cleaned = self.sanitize_sentence(sent)
            # Re-verify density after cleaning (e.g. if the whole sentence was just "welcome")
            if cleaned and self.is_concept_dense(cleaned):
                sanitized.append(cleaned)
                
        # Re-join sentences properly
        result = " ".join(sanitized)
        # Fix multiple spaces
        result = re.sub(r'\s+', ' ', result)
        # Fix space before punctuation
        result = re.sub(r'\s+([.,!?])', r'\1', result)
        
        return result.strip()
