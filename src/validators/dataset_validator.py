"""Stage 11: Validation.

Runs a battery of structural checks over the final instruction dataset
before export: missing fields, empty outputs, invalid JSON-serializability,
encoding problems, duplicate IDs, and invalid categories. Raises
`DatasetValidationError` only for issues severe enough that export should
be blocked; everything else is collected into a report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Set

from src.processors.models import InstructionExample
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = ("id", "instruction", "input", "output", "category", "difficulty", "source")
VALID_CATEGORIES = {
    "agent_engineering",
    "llm_engineering",
    "retrieval_systems",
    "data_science",
    "ai_system_design",
    "technical_project_management",
    "uncategorized",
}
VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}


class DatasetValidationError(Exception):
    """Raised when the dataset fails validation severely enough to block export."""


@dataclass
class ValidationIssue:
    example_id: str
    issue_type: str
    detail: str


@dataclass
class ValidationReport:
    total_examples: int = 0
    valid_examples: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        by_type: dict = {}
        for issue in self.issues:
            by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1
        return {
            "total_examples": self.total_examples,
            "valid_examples": self.valid_examples,
            "invalid_examples": self.total_examples - self.valid_examples,
            "issues_by_type": by_type,
            "issues": [
                {"example_id": i.example_id, "issue_type": i.issue_type, "detail": i.detail}
                for i in self.issues
            ],
        }


class DatasetValidator:
    def __init__(self, strict: bool = False) -> None:
        """`strict=True` raises `DatasetValidationError` if any invalid
        example is found rather than just filtering it out.
        """
        self.strict = strict

    def validate(self, examples: List[InstructionExample]) -> tuple[List[InstructionExample], ValidationReport]:
        report = ValidationReport(total_examples=len(examples))
        seen_ids: Set[str] = set()
        valid_examples: List[InstructionExample] = []

        for example in examples:
            issues = self._validate_example(example, seen_ids)
            if issues:
                report.issues.extend(issues)
                if self.strict:
                    raise DatasetValidationError(
                        f"Validation failed for example {example.id}: {[i.issue_type for i in issues]}"
                    )
                continue
            seen_ids.add(example.id)
            valid_examples.append(example)

        report.valid_examples = len(valid_examples)
        logger.info(
            f"Validation complete: {report.valid_examples}/{report.total_examples} example(s) valid "
            f"({len(report.issues)} issue(s) found)."
        )
        return valid_examples, report

    def _validate_example(self, example: InstructionExample, seen_ids: Set[str]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        data = example.to_dict()

        for field_name in REQUIRED_FIELDS:
            value = data.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                if field_name == "input":
                    continue  # input is allowed to be empty by schema
                issues.append(ValidationIssue(example.id, "missing_field", field_name))

        if not example.output.strip():
            issues.append(ValidationIssue(example.id, "empty_output", "output field is empty"))

        if example.id in seen_ids:
            issues.append(ValidationIssue(example.id, "duplicate_id", "duplicate example id"))

        if example.category not in VALID_CATEGORIES:
            issues.append(ValidationIssue(example.id, "invalid_category", example.category))

        if example.difficulty not in VALID_DIFFICULTIES:
            issues.append(ValidationIssue(example.id, "invalid_difficulty", example.difficulty))

        try:
            json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            issues.append(ValidationIssue(example.id, "invalid_json", str(exc)))

        try:
            example.instruction.encode("utf-8")
            example.output.encode("utf-8")
        except UnicodeEncodeError as exc:
            issues.append(ValidationIssue(example.id, "encoding_error", str(exc)))

        return issues
