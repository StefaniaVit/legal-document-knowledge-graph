"""Unit tests for src/ingestion/chunker.py's structure-aware splitting logic."""
from src.ingestion.chunker import _chunk_unit, _split_by_sentences, _split_by_subparagraphs


class TestSplitBySubparagraphs:
    def test_splits_on_bare_numbering(self):
        text = "1. First paragraph here. 2. Second paragraph here. 3. Third paragraph here."
        parts = _split_by_subparagraphs(text)
        assert len(parts) == 3
        assert parts[0].startswith("1. First")
        assert parts[2].startswith("3. Third")

    def test_splits_on_parenthesized_numbering(self):
        # Real pattern from NIS2 Article 6 (a definitions article)
        text = (
            "(1) 'network and information system' means an electronic communications network. "
            "(2) 'security of network and information systems' means the ability to resist. "
            "(3) 'cybersecurity' means cybersecurity as defined in Article 2."
        )
        parts = _split_by_subparagraphs(text)
        assert len(parts) == 3
        assert parts[0].startswith("(1)")
        assert parts[1].startswith("(2)")

    def test_single_paragraph_is_not_split(self):
        # Only one numbered marker -- not a real sequence, leave untouched
        text = "1. This is the only paragraph, there is no second one."
        assert _split_by_subparagraphs(text) == [text]

    def test_broken_sequence_is_not_split(self):
        # Numbers present but not a clean 1,2,3.. sequence (e.g. cross-references
        # to other paragraphs mixed into the text) -- falls back to no split
        text = "See paragraph 5 for details. Also refer to paragraph 9 above for context."
        assert _split_by_subparagraphs(text) == [text]

    def test_no_numbering_at_all_is_not_split(self):
        text = "Plain prose with no numbered structure whatsoever in it."
        assert _split_by_subparagraphs(text) == [text]


class TestSplitBySentences:
    def test_splits_on_sentence_boundaries(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = _split_by_sentences(text, max_chars=30)
        assert len(chunks) > 1
        assert all(len(c) <= 40 for c in chunks)  # some slack for join spacing

    def test_does_not_split_on_legal_abbreviations(self):
        # "No. 45" and "p. 20" must not be treated as sentence boundaries --
        # the char after the period is a digit, not a capital letter.
        text = "See Commission Recommendation No. 45 of 2003 concerning enterprises, p. 20 for details."
        chunks = _split_by_sentences(text, max_chars=10000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_packs_short_sentences_together_up_to_max_chars(self):
        text = "One. Two. Three. Four."
        chunks = _split_by_sentences(text, max_chars=10000)
        assert len(chunks) == 1


class TestChunkUnit:
    def test_short_text_is_returned_as_single_chunk(self):
        text = "A short recital that doesn't need splitting at all."
        assert _chunk_unit(text, max_chars=2000) == [text]

    def test_long_text_with_clean_structure_splits_by_subparagraph(self):
        para = "This is a reasonably long paragraph of legal text. " * 5
        text = "".join(f"{i}. {para}" for i in range(1, 4))
        # max_chars set above each individual subparagraph's own length (~260 chars)
        # so this isolates the subparagraph split itself, not the further
        # per-part sentence-splitting fallback (covered by a separate test below).
        parts = _chunk_unit(text, max_chars=300)
        assert len(parts) == 3
        assert parts[0].startswith("1.")

    def test_long_text_with_no_structure_falls_back_to_sentences(self):
        text = ("This is one sentence in a long recital with no numbered structure at all. " * 10)
        parts = _chunk_unit(text, max_chars=200)
        assert len(parts) > 1
        # every part should respect the limit reasonably (small slack for join spacing)
        assert all(len(p) <= 250 for p in parts)

    def test_oversized_subparagraph_gets_further_split_by_sentences(self):
        # A clean 1,2,3.. structure exists, but one individual sub-paragraph
        # is itself too long -- that one part alone should get sentence-split.
        short_para = "Short paragraph text here."
        long_para = "This sentence repeats to make the paragraph very long indeed. " * 8
        text = f"1. {short_para} 2. {long_para} 3. {short_para}"
        parts = _chunk_unit(text, max_chars=150)
        assert len(parts) > 3  # the long part 2 got split into multiple pieces
