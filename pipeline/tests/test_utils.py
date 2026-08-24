from src.utils.hashing import (
    document_id,
    hamming_distance,
    simhash,
    simhash_similarity,
    stable_hash,
)
from src.utils.text_utils import (
    code_character_ratio,
    extract_code_blocks,
    normalize_whitespace,
    remove_boilerplate_lines,
    simple_readability_score,
    strip_code_blocks,
)
from src.utils.tokenizer import count_tokens, truncate_to_tokens


def test_stable_hash_is_deterministic():
    assert stable_hash("hello world") == stable_hash("hello world")
    assert stable_hash("hello world") != stable_hash("hello there")


def test_document_id_is_namespaced_and_deterministic():
    id_a = document_id("github", "repo/path.md")
    id_b = document_id("github", "repo/path.md")
    id_c = document_id("documentation", "repo/path.md")
    assert id_a == id_b
    assert id_a.startswith("github_")
    assert id_a != id_c


def test_simhash_similarity_identical_text_is_one():
    text = "LangGraph is a library for building stateful multi-agent workflows."
    hash_a = simhash(text)
    hash_b = simhash(text)
    assert hamming_distance(hash_a, hash_b) == 0
    assert simhash_similarity(hash_a, hash_b) == 1.0


def test_simhash_similarity_different_text_is_lower():
    hash_a = simhash("LangGraph is a library for building stateful multi-agent workflows.")
    hash_b = simhash("Quantum computing uses superposition and entanglement for computation.")
    assert simhash_similarity(hash_a, hash_b) < 0.9


def test_extract_code_blocks_finds_language_and_code():
    markdown = "Some text\n```python\nprint('hi')\n```\nmore text"
    blocks = extract_code_blocks(markdown)
    assert len(blocks) == 1
    assert blocks[0][0] == "python"
    assert "print('hi')" in blocks[0][1]


def test_strip_code_blocks_removes_fences():
    markdown = "Intro\n```python\nx = 1\n```\nOutro"
    stripped = strip_code_blocks(markdown)
    assert "x = 1" not in stripped
    assert "Intro" in stripped and "Outro" in stripped


def test_code_character_ratio_bounds():
    text_no_code = "Just plain prose with no code at all."
    assert code_character_ratio(text_no_code) == 0.0

    text_all_code = "```python\n" + ("x = 1\n" * 50) + "```"
    ratio = code_character_ratio(text_all_code)
    assert 0.0 < ratio <= 1.0


def test_remove_boilerplate_lines_strips_cookie_banner():
    text = "We use cookies to improve your experience.\nActual useful content here."
    cleaned = remove_boilerplate_lines(text)
    assert "cookies" not in cleaned.lower()
    assert "Actual useful content" in cleaned


def test_normalize_whitespace_collapses_blank_lines():
    text = "Line one\n\n\n\n\nLine two"
    normalized = normalize_whitespace(text)
    assert "\n\n\n" not in normalized


def test_simple_readability_score_range():
    score = simple_readability_score("This is a short, simple sentence. It is easy to read.")
    assert 0.0 <= score <= 1.0


def test_count_tokens_nonzero_for_text():
    assert count_tokens("") == 0
    assert count_tokens("hello world, this is a test sentence.") > 0


def test_truncate_to_tokens_respects_budget():
    long_text = "word " * 500
    truncated = truncate_to_tokens(long_text, max_tokens=10)
    assert count_tokens(truncated) <= 15  # small slack for encoding boundary effects
