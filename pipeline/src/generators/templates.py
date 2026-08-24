"""Stage 9: Instruction templates.

Each template is a pure function `(Chunk, DocumentMetadata) -> List[InstructionExample-ish dict]`
that deterministically derives instruction/input/output triples from a
chunk's text and structure. Templates are rule-based (no external LLM
call) so the resulting dataset is fully reproducible from the collected
corpus alone -- a hard requirement for the research use case described
in the project brief.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.processors.models import Chunk, DocumentMetadata
from src.utils.text_utils import extract_code_blocks, strip_code_blocks

_QUESTION_HEADING_HINTS = ("how", "why", "what", "when", "should")


def _difficulty_for_chunk(chunk: Chunk) -> str:
    token_count = chunk.token_count
    if token_count < 150:
        return "beginner"
    if token_count < 500:
        return "intermediate"
    return "advanced"


def _title_from_heading_path(chunk: Chunk) -> str:
    return chunk.heading_path[-1] if chunk.heading_path else "this topic"


def qa_template(chunk: Chunk, metadata: DocumentMetadata) -> List[Dict]:
    """Question-answer pairs derived from heading + body text."""
    if not chunk.heading_path:
        return []

    title = _title_from_heading_path(chunk)
    body = strip_code_blocks(chunk.text).strip()
    if len(body) < 40:
        return []

    lower_title = title.lower()
    if any(hint in lower_title for hint in _QUESTION_HEADING_HINTS):
        question = title if title.endswith("?") else f"{title}?"
    else:
        question = f"What is {title} and how is it used in the context of {metadata.framework or metadata.category}?"

    return [
        {
            "instruction": question,
            "input": "",
            "output": body,
            "difficulty": _difficulty_for_chunk(chunk),
            "template": "qa",
        }
    ]


def instruction_following_template(chunk: Chunk, metadata: DocumentMetadata) -> List[Dict]:
    """Direct 'explain / describe / summarize' instructions."""
    title = _title_from_heading_path(chunk)
    body = strip_code_blocks(chunk.text).strip()
    if len(body) < 60:
        return []

    instruction = f"Explain {title} in the context of {metadata.framework or metadata.category}, including key concepts and practical considerations."
    return [
        {
            "instruction": instruction,
            "input": "",
            "output": body,
            "difficulty": _difficulty_for_chunk(chunk),
            "template": "instruction_following",
        }
    ]


def code_generation_template(chunk: Chunk, metadata: DocumentMetadata) -> List[Dict]:
    """Code + explanation pairs, framed as 'write code that does X'."""
    code_blocks = extract_code_blocks(chunk.text)
    if not code_blocks:
        return []

    examples: List[Dict] = []
    title = _title_from_heading_path(chunk)
    prose = strip_code_blocks(chunk.text).strip()

    for language, code in code_blocks:
        if len(code.strip()) < 20 or language in ("text", ""):
            continue
        context_hint = prose[:400] if prose else f"functionality related to {title}"
        instruction = (
            f"Write {language} code that demonstrates {title} "
            f"({metadata.framework or metadata.category})."
        )
        input_text = f"Context: {context_hint}" if prose else ""
        output = f"```{language}\n{code}\n```"
        examples.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "difficulty": _difficulty_for_chunk(chunk),
                "template": "code_generation",
            }
        )
    return examples


def architecture_explanation_template(chunk: Chunk, metadata: DocumentMetadata) -> List[Dict]:
    """Framed around system/architecture-level reasoning."""
    body = strip_code_blocks(chunk.text).strip()
    architecture_keywords = ("architecture", "design", "system", "workflow", "pipeline", "structure")
    combined_text = (chunk.text + " ".join(chunk.heading_path)).lower()
    if not any(keyword in combined_text for keyword in architecture_keywords):
        return []
    if len(body) < 80:
        return []

    title = _title_from_heading_path(chunk)
    instruction = f"Describe the architectural design and component interactions involved in {title}."
    return [
        {
            "instruction": instruction,
            "input": "",
            "output": body,
            "difficulty": "advanced",
            "template": "architecture_explanation",
        }
    ]


def debugging_template(chunk: Chunk, metadata: DocumentMetadata) -> List[Dict]:
    """Framed around troubleshooting / common pitfalls when such language is present."""
    combined_lower = chunk.text.lower()
    debug_keywords = ("error", "exception", "fails", "issue", "bug", "troubleshoot", "fix", "raise ")
    if not any(keyword in combined_lower for keyword in debug_keywords):
        return []

    body = strip_code_blocks(chunk.text).strip()
    if len(body) < 60:
        return []

    title = _title_from_heading_path(chunk)
    instruction = f"A developer working with {title} ({metadata.framework or metadata.category}) is encountering issues. Explain likely causes and how to resolve them."
    return [
        {
            "instruction": instruction,
            "input": "",
            "output": body,
            "difficulty": "advanced",
            "template": "debugging",
        }
    ]


def reasoning_template(chunk: Chunk, metadata: DocumentMetadata) -> List[Dict]:
    """Framed around 'why' / tradeoff reasoning rather than pure recall."""
    combined_lower = chunk.text.lower()
    reasoning_keywords = ("because", "trade-off", "tradeoff", "advantage", "disadvantage", "however", "in contrast")
    if not any(keyword in combined_lower for keyword in reasoning_keywords):
        return []

    body = strip_code_blocks(chunk.text).strip()
    if len(body) < 80:
        return []

    title = _title_from_heading_path(chunk)
    instruction = f"What are the key trade-offs and reasoning behind design decisions related to {title}?"
    return [
        {
            "instruction": instruction,
            "input": "",
            "output": body,
            "difficulty": "advanced",
            "template": "reasoning",
        }
    ]


def comparison_template(chunk: Chunk, metadata: DocumentMetadata) -> List[Dict]:
    """Framed around comparing two or more named entities in the text."""
    vs_pattern = re.compile(r"\b([A-Z][\w\.\-]{1,30})\s+(?:vs\.?|versus|compared to)\s+([A-Z][\w\.\-]{1,30})", re.IGNORECASE)
    match = vs_pattern.search(chunk.text)
    if not match:
        return []

    body = strip_code_blocks(chunk.text).strip()
    if len(body) < 60:
        return []

    entity_a, entity_b = match.group(1), match.group(2)
    instruction = f"Compare {entity_a} and {entity_b} in the context of {metadata.framework or metadata.category}, covering their key differences and appropriate use cases."
    return [
        {
            "instruction": instruction,
            "input": "",
            "output": body,
            "difficulty": "advanced",
            "template": "comparison",
        }
    ]


TEMPLATE_REGISTRY = {
    "qa": qa_template,
    "instruction_following": instruction_following_template,
    "code_generation": code_generation_template,
    "architecture_explanation": architecture_explanation_template,
    "debugging": debugging_template,
    "reasoning": reasoning_template,
    "comparison": comparison_template,
}


def get_template(name: str) -> Optional[callable]:
    return TEMPLATE_REGISTRY.get(name)
