import re
from collections import Counter
from typing import List, Set

try:
    import spacy
    # Load basic language model, disable heavy pipelines for speed
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except (ImportError, Exception):
    nlp = None


# ── OCR Symbol & Artefact Patterns ──
# These patterns are dynamic — no hardcoded domain words.
# They match symbols, encoding artefacts, and formatting noise that OCR produces.

# Unicode replacement characters and common OCR encoding errors
_OCR_UNICODE_NOISE = re.compile(
    r'[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]'
    r'|[\u00b7\u2022\u2023\u2043\u204c\u204d]'   # various bullet chars
    r'|[\u2018\u2019\u201a\u201b]'                # fancy single quotes → keep as '
    r'|[\u201c\u201d\u201e\u201f]'                # fancy double quotes → keep as "
    r'|[\u2013\u2014\u2015]'                      # em/en dashes → keep as -
    r'|[\u00ad]'                                   # soft hyphen
)

# OCR symbol noise in slide text: ?, →, arrows, box-drawing chars
_OCR_SYMBOL_NOISE = re.compile(
    r'[\u2500-\u257f]'      # Box drawing
    r'|[\u2580-\u259f]'     # Block elements
    r'|[\u25a0-\u25ff]'     # Geometric shapes
    r'|[\u2600-\u26ff]'     # Misc symbols
    r'|[\u2190-\u21ff]'     # Arrows (unicode)
    r'|→|←|↑|↓|↔|↕'        # Common arrows
)

# OCR line-noise patterns: isolated letters/numbers on their own line,
# headers/footers repeated across slides, slide numbers, etc.
_OCR_LINE_NOISE = re.compile(
    r'^[\s\d\.\,\;\:\!\@\#\$\%\^\&\*\(\)\[\]\{\}\/\\\|\-\_\=\+\~\`\'\"]{1,5}$'
)

# Question-mark artefact patterns (KG separator leaked into OCR text)
_Q_IN_OCR = re.compile(r'(?<!\w)\?(?!\w)|(?<=\s)\?(?=\s)|\?$|^\?')


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
            # Change 16: Filler phrase removal
            r"(?i)^(so basically|you know|kind of|now here|coming to|as I said|moving on(?: to)?)\s*[,.]?\s*",
            r"(?i)^(okay so|alright so|right so|well basically)\s*[,.]?\s*",
        ]
        self.repeated_ocr_noise = set()
        
        # Change 17: Contraction expansion dict
        self._contractions = {
            "it's": "it is", "don't": "do not", "doesn't": "does not",
            "didn't": "did not", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "won't": "will not",
            "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
            "can't": "cannot", "hasn't": "has not", "haven't": "have not",
            "hadn't": "had not", "they're": "they are", "we're": "we are",
            "you're": "you are", "he's": "he is", "she's": "she is",
            "that's": "that is", "there's": "there is", "what's": "what is",
            "who's": "who is", "let's": "let us", "i'm": "I am",
            "we'll": "we will", "they'll": "they will", "you'll": "you will",
            "i'll": "I will", "i've": "I have", "we've": "we have",
            "they've": "they have", "you've": "you have",
        }
        
        # Change 18: Number word normalization patterns
        self._number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'thirty': '30', 'forty': '40', 'fifty': '50',
            'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
            'hundred': '00', 'thousand': '000',
        }

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text (Change 17)."""
        # Case-insensitive contraction replacement
        words = text.split()
        expanded = []
        for word in words:
            lower = word.lower().rstrip('.,!?;:')
            trailing = word[len(lower):]  # Preserve trailing punctuation
            if lower in self._contractions:
                expanded.append(self._contractions[lower] + trailing)
            else:
                expanded.append(word)
        return ' '.join(expanded)

    def _normalize_number_words(self, text: str) -> str:
        """Normalize spoken number words to digits (Change 18)."""
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            lower = words[i].lower().rstrip('.,!?;:')
            trailing = words[i][len(lower):]
            
            if lower in self._number_words:
                num_parts = [self._number_words[lower]]
                j = i + 1
                while j < len(words):
                    next_lower = words[j].lower().rstrip('.,!?;:')
                    if next_lower in self._number_words:
                        num_parts.append(self._number_words[next_lower])
                        j += 1
                    else:
                        break
                
                if len(num_parts) > 1:
                    combined = ''.join(num_parts)
                    result.append(combined + trailing)
                else:
                    result.append(self._number_words[lower] + trailing)
                i = j
            else:
                result.append(words[i])
                i += 1
        
        return ' '.join(result)

    def clean_ocr_symbols(self, text: str) -> str:
        """
        Remove OCR-introduced symbol artefacts from slide text.
        Dynamically handles:
          - Unicode replacement and noise characters
          - Box-drawing / geometric shape characters
          - Arrow symbols (→ ← ↑ ↓)
          - Standalone ? question-mark artefacts (KG separator leaked from OCR)
          - PDF stream operators leaked into extracted text (BT, ET, Td, Tj, etc.)
          - Repeated punctuation / symbol sequences
        No hardcoded domain words — purely structural pattern matching.
        """
        if not text:
            return text

        # 0. Strip PDF stream operators that leak during PDF text extraction
        #    e.g. "0.000 g BT 59.53 735.83 Td *e" — these are PDF graphics operators
        #    Pattern: standalone uppercase 1-3 char tokens after numbers, or known operators
        text = re.sub(r'\b(?:BT|ET|Td|TD|Tj|TJ|Tf|Tm|T\*)\b', ' ', text)
        text = re.sub(r'\b\d+\.\d+\s+\d+\.\d+\s+Td\b', ' ', text)  # "59.53 735.83 Td"
        text = re.sub(r'\b\d+\.\d+\s+[gGrRkKcs]\b', ' ', text)     # "0.000 g"
        text = re.sub(r'\s\*[a-z]\b', ' ', text)                    # "*e", "*f"

        # 1. Replace fancy quotes with ASCII equivalents (preserve meaning)
        text = re.sub(r'[\u2018\u2019\u201a\u201b]', "'", text)
        text = re.sub(r'[\u201c\u201d\u201e\u201f]', '"', text)
        # em/en dashes → hyphen
        text = re.sub(r'[\u2013\u2014\u2015]', '-', text)

        # 2. Remove pure OCR noise symbols (box drawing, geometric, arrows)
        text = _OCR_SYMBOL_NOISE.sub(' ', text)

        # 3. Remove unicode replacement / control characters
        text = re.sub(
            r'[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u00ad]',
            '', text
        )

        # 4. Remove standalone ? that are KG separator artefacts, NOT real questions.
        text = re.sub(r'\s\?\s', ' ', text)
        text = re.sub(r'^\?+\s*', '', text)
        text = re.sub(r'\s*\?+$', '', text)
        text = re.sub(r'([^a-zA-Z])\?', r'\1', text)
        text = re.sub(r'([A-Za-z])\?([A-Z])', r'\1 \2', text)
        text = text.replace('?', '')

        # 5. Remove repeated punctuation (OCR sometimes produces "....." or "-----")
        text = re.sub(r'([.\-_=*#@])\1{2,}', '', text)

        # 6. Collapse whitespace
        text = re.sub(r'\s{2,}', ' ', text).strip()

        return text

    def clean_ocr_line(self, line: str) -> str:
        """
        Determine if a single OCR line is pure noise (should be dropped).
        Returns cleaned line or empty string if noise.
        """
        stripped = line.strip()
        if not stripped:
            return ''
        # Drop lines that are only symbols/numbers/punctuation (OCR line noise)
        if _OCR_LINE_NOISE.match(stripped):
            return ''
        # Drop lines shorter than 3 chars (almost always OCR artefact)
        if len(stripped) < 3:
            return ''
        # Apply symbol cleanup
        cleaned = self.clean_ocr_symbols(stripped)
        # If after cleanup nothing meaningful remains, drop
        if len(cleaned.split()) < 1:
            return ''
        return cleaned

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
        
        # 0a. OCR symbol cleanup (NEW) — remove ?, →, box chars, unicode noise
        s = self.clean_ocr_symbols(s)

        # 0b. Expand contractions (Change 17)
        s = self._expand_contractions(s)
        
        # 0c. Normalize number words (Change 18)
        s = self._normalize_number_words(s)
        
        # 1. OCR specific dynamic noise filter (watermarks, slide headers)
        for noise_line in self.repeated_ocr_noise:
            if noise_line in s:
                s = s.replace(noise_line, "")
                
        # 2. Heuristic regex filter for standard tutorial noise + filler phrases (Change 16)
        for pattern in self.framing_patterns:
            s = re.sub(pattern, "", s)
        
        # Final: collapse spaces and strip
        s = re.sub(r'\s{2,}', ' ', s).strip()
        return s

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize a full paragraph/document of text.
        Includes OCR symbol removal, contraction expansion, and framing-phrase removal.
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
            # Re-verify density after cleaning
            if cleaned and self.is_concept_dense(cleaned):
                sanitized.append(cleaned)
                
        # Re-join sentences properly
        result = " ".join(sanitized)
        # Fix multiple spaces
        result = re.sub(r'\s+', ' ', result)
        # Fix space before punctuation
        result = re.sub(r'\s+([.,!?])', r'\1', result)
        # Final nuclear ? removal (safety net)
        result = result.replace('?', '')
        result = re.sub(r'\s{2,}', ' ', result).strip()
        
        return result.strip()

    def sanitize_slide_text(self, text: str) -> str:
        """
        Sanitize raw OCR slide text (before KG extraction).
        More aggressive than sanitize_text — removes line-level noise too.
        """
        if not text:
            return ""
        lines = text.split('\n')

        # Pre-pass: repair OCR line-break word splits BEFORE noise filtering.
        # e.g. "Oper\nations" → "Operations" when OCR splits a word across lines.
        # Signal: line ends with lowercase (no trailing punctuation) AND
        #         next line starts with lowercase (continuation of same word).
        joined_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            next_line = lines[i+1].strip() if i + 1 < len(lines) else ""
            if (stripped_line
                    and next_line
                    and not stripped_line[-1] in '.!?:;,'
                    and re.search(r'[a-z]$', stripped_line)
                    and re.match(r'^[a-z]', next_line)
                    and len(stripped_line) <= 20):  # Only merge short fragments
                merged = stripped_line + next_line
                joined_lines.append(merged)
                i += 2
                continue
            joined_lines.append(line)
            i += 1

        cleaned_lines = []
        for line in joined_lines:
            cl = self.clean_ocr_line(line)
            if cl:
                cleaned_lines.append(cl)
        result = '\n'.join(cleaned_lines)
        return result