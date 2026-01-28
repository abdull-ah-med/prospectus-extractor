import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from .chunker import TextChunk

logger = logging.getLogger(__name__)


class SectionLabel(str, Enum):
    DEPARTMENTS = "departments"
    PROGRAMS = "programs"
    CURRICULUM = "curriculum"
    FEES = "fees"
    ADMISSIONS = "admissions"
    FACILITIES = "facilities"
    CONTACT = "contact"
    GENERAL = "general"


@dataclass
class ClassificationResult:
    chunk: TextChunk
    label: SectionLabel
    confidence: float
    matched_patterns: List[str]


class ClassifierService:
    SECTION_PATTERNS: Dict[SectionLabel, List[str]] = {
        SectionLabel.DEPARTMENTS: [
            r"(?i)department\s+of\s+[a-z\s&]+",
            r"(?i)faculty\s+of\s+[a-z\s&]+",
            r"(?i)school\s+of\s+[a-z\s&]+",
            r"(?i)college\s+of\s+[a-z\s&]+",
            r"(?i)faculties\s*(&|and)\s*departments",
            r"(?i)academic\s+departments",
        ],
        SectionLabel.PROGRAMS: [
            r"(?i)\b(bs|b\.s\.|bachelor|bsc|b\.sc)\b",
            r"(?i)\b(ms|m\.s\.|master|msc|m\.sc|mba|m\.b\.a)\b",
            r"(?i)\b(phd|ph\.d|doctorate|doctoral)\b",
            r"(?i)\b(degree|program|programmes?|courses?)\b.*offered",
            r"(?i)undergraduate\s+programs?",
            r"(?i)graduate\s+programs?",
            r"(?i)postgraduate\s+programs?",
        ],
        SectionLabel.CURRICULUM: [
            r"(?i)\bcredit\s+hours?\b",
            r"(?i)\bsemester\s+(1|2|3|4|5|6|7|8|i|ii|iii|iv|v|vi|vii|viii)\b",
            r"(?i)\bcourse\s+code\b",
            r"(?i)\bsyllabus\b",
            r"(?i)\bcurriculum\b",
            r"(?i)\bprerequisites?\b",
            r"(?i)\belective\s+courses?\b",
            r"(?i)\bcore\s+courses?\b",
        ],
        SectionLabel.FEES: [
            r"(?i)\bfee\s+structure\b",
            r"(?i)\btuition\s+fees?\b",
            r"(?i)\bsemester\s+fees?\b",
            r"(?i)\badmission\s+fees?\b",
            r"(?i)\bpkr\s*[\d,]+",
            r"(?i)rs\.?\s*[\d,]+",
            r"(?i)\bpayment\b.*\bdue\b",
            r"(?i)\bfinancial\s+aid\b",
            r"(?i)\bscholarship\b",
        ],
        SectionLabel.ADMISSIONS: [
            r"(?i)\badmission\s+(criteria|requirements?|process|procedure)\b",
            r"(?i)\beligibility\s+(criteria|requirements?)\b",
            r"(?i)\bapplication\s+(deadline|form|process)\b",
            r"(?i)\bentry\s+test\b",
            r"(?i)\binterview\s+schedule\b",
            r"(?i)\bmerit\s+list\b",
            r"(?i)\bhow\s+to\s+apply\b",
            r"(?i)\brequired\s+documents?\b",
            r"(?i)\bminimum\s+marks?\b",
        ],
        SectionLabel.FACILITIES: [
            r"(?i)\blaborator(y|ies)\b",
            r"(?i)\blibrar(y|ies)\b",
            r"(?i)\bhostel\b",
            r"(?i)\bdormitor(y|ies)\b",
            r"(?i)\bsports\s+complex\b",
            r"(?i)\bgym(nasium)?\b",
            r"(?i)\bcafeteria\b",
            r"(?i)\bauditorium\b",
            r"(?i)\bresearch\s+center\b",
            r"(?i)\bcomputer\s+lab\b",
            r"(?i)\bwi-?fi\b",
            r"(?i)\bparking\b",
            r"(?i)\bcampus\s+facilities\b",
        ],
        SectionLabel.CONTACT: [
            r"(?i)\bcontact\s+us\b",
            r"(?i)\baddress\s*:",
            r"(?i)\bphone\s*:",
            r"(?i)\bemail\s*:",
            r"(?i)\bwebsite\s*:",
            r"(?i)\bsocial\s+media\b",
            r"(?i)\bfollow\s+us\b",
        ],
    }

    SECTION_KEYWORDS: Dict[SectionLabel, List[str]] = {
        SectionLabel.DEPARTMENTS: ["department", "faculty", "school", "college", "dean"],
        SectionLabel.PROGRAMS: ["degree", "bachelor", "master", "phd", "program", "course", "diploma"],
        SectionLabel.CURRICULUM: ["credit", "semester", "syllabus", "prerequisite", "elective", "course code"],
        SectionLabel.FEES: ["fee", "tuition", "payment", "rupee", "pkr", "scholarship", "financial"],
        SectionLabel.ADMISSIONS: ["admission", "eligibility", "apply", "deadline", "merit", "test", "interview"],
        SectionLabel.FACILITIES: ["lab", "library", "hostel", "sports", "cafeteria", "auditorium", "gym"],
        SectionLabel.CONTACT: ["contact", "address", "phone", "email", "website"],
    }

    def __init__(self):
        self._compiled_patterns: Dict[SectionLabel, List[re.Pattern]] = {}
        for label, patterns in self.SECTION_PATTERNS.items():
            self._compiled_patterns[label] = [re.compile(p) for p in patterns]

    def classify_chunk(self, chunk: TextChunk) -> ClassificationResult:
        text = chunk.text
        scores: Dict[SectionLabel, float] = {}
        matched: Dict[SectionLabel, List[str]] = {}

        for label, patterns in self._compiled_patterns.items():
            score = 0.0
            matches = []
            for pattern in patterns:
                if pattern.search(text):
                    score += 2.0
                    matches.append(pattern.pattern)

            keywords = self.SECTION_KEYWORDS.get(label, [])
            text_lower = text.lower()
            for kw in keywords:
                if kw in text_lower:
                    score += 0.5
                    if kw not in matches:
                        matches.append(f"keyword:{kw}")

            scores[label] = score
            matched[label] = matches

        if chunk.section_label:
            for label in SectionLabel:
                if chunk.section_label.lower() == label.value:
                    scores[label] += 3.0
                    break

        best_label = SectionLabel.GENERAL
        best_score = 0.0
        for label, score in scores.items():
            if score > best_score:
                best_score = score
                best_label = label

        max_possible = 10.0
        confidence = min(best_score / max_possible, 1.0)

        return ClassificationResult(
            chunk=chunk,
            label=best_label,
            confidence=confidence,
            matched_patterns=matched.get(best_label, [])
        )

    def classify_chunks(self, chunks: List[TextChunk]) -> Dict[SectionLabel, List[TextChunk]]:
        classified: Dict[SectionLabel, List[TextChunk]] = {label: [] for label in SectionLabel}

        for chunk in chunks:
            result = self.classify_chunk(chunk)
            chunk.section_label = result.label.value
            classified[result.label].append(chunk)

        for label, section_chunks in classified.items():
            if section_chunks:
                logger.info(f"Section '{label.value}': {len(section_chunks)} chunks")

        return classified

    def get_chunks_by_section(
        self, 
        chunks: List[TextChunk], 
        section: SectionLabel,
        min_confidence: float = 0.0
    ) -> List[TextChunk]:
        result = []
        for chunk in chunks:
            classification = self.classify_chunk(chunk)
            if classification.label == section and classification.confidence >= min_confidence:
                result.append(chunk)
        return result


classifier_service = ClassifierService()
