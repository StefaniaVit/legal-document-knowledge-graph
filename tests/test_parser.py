"""
Unit tests for src/ingestion/parser.py's pure-function logic (footer
stripping, title/date extraction, footnote-block stripping, article
extraction). Deliberately excludes _extract_recitals(), since that calls
the local LLM via recital_classifier.is_recital() -- not something a fast,
deterministic unit test suite should depend on.
"""
from src.ingestion.parser import (
    _extract_articles,
    _extract_title_and_date,
    _is_footer_line,
    _strip_embedded_footnote_blocks,
)


class TestIsFooterLine:
    def test_matches_en_marker(self):
        assert _is_footer_line("EN")

    def test_matches_official_journal_line(self):
        assert _is_footer_line("Official Journal of the European Union")

    def test_matches_date_line(self):
        assert _is_footer_line("12.7.2024")

    def test_matches_page_ref_line(self):
        assert _is_footer_line("L 1689/1")
        assert _is_footer_line("L  1689/1")  # internal spacing variant

    def test_does_not_match_body_text(self):
        assert not _is_footer_line("This Regulation applies to processing of personal data.")

    def test_does_not_match_partial_match(self):
        # regexes are anchored (^...$) -- footer text embedded in a longer
        # line must not trigger a false positive
        assert not _is_footer_line("As published in the Official Journal of the European Union today")


class TestExtractTitleAndDate:
    def test_extracts_gdpr_style_header(self):
        text = (
            "I \n(Legislative acts) \nREGULATIONS \n"
            "REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL \n"
            "of 27 April 2016 \n"
            "on the protection of natural persons with regard to the processing of personal data "
            "(General Data Protection Regulation) \n"
            "(Text with EEA relevance) \n"
            "THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,"
        )
        title, date = _extract_title_and_date(text)
        assert "REGULATION (EU) 2016/679" in title
        assert "General Data Protection Regulation" in title
        assert date == "27 April 2016"

    def test_returns_empty_when_no_anchor_found(self):
        title, date = _extract_title_and_date("Some unrelated text with no act header.")
        assert title == ""
        assert date == ""


class TestStripEmbeddedFootnoteBlocks:
    def test_no_footnote_block_returns_text_unchanged(self):
        text = "This Regulation applies to all Member States equally and without exception."
        assert _strip_embedded_footnote_blocks(text) == text

    def test_single_inline_reference_is_left_alone(self):
        # A lone "(N)" is normal prose (an inline footnote reference), not a
        # footnote-definition block -- only a run of 2+ triggers stripping.
        text = "The Council of the European Union approved this measure (5) after due consideration."
        assert _strip_embedded_footnote_blocks(text) == text

    def test_strips_block_that_interrupts_mid_sentence(self):
        # Real case from GDPR recital 17: the footnote block lands between
        # "Personal or" and "household activities", so resumption is
        # lowercase -- the OJ-ending pattern must be used to find the
        # boundary, not a capitalized-sentence-start heuristic.
        text = (
            "Personal or (1) Commission Recommendation of 6 May 2003 concerning the definition "
            "of micro, small and medium‑sized enterprises (C(2003) 1422) "
            "(OJ L 124, 20.5.2003, p. 36). "
            "(2) Regulation (EC) No 45/2001 of the European Parliament and of the Council of "
            "18 December 2000 on the protection of individuals with regard to the processing "
            "of personal data by the Community institutions and bodies and on the free movement "
            "of such data (OJ L 8, 12.1.2001, p. 1). "
            "household activities could include correspondence and the holding of addresses."
        )
        result = _strip_embedded_footnote_blocks(text)
        assert result == "Personal or household activities could include correspondence and the holding of addresses."

    def test_strips_block_at_sentence_boundary(self):
        # Real case from Data Governance Act recital 2: footnote block
        # happens to land right at a sentence boundary, so the fallback
        # capitalized-sentence-start heuristic also needs to work.
        text = (
            "could be pivotal for the rapid development of artificial intelligence technologies. "
            "(1) OJ C 286, 16.7.2021, p. 38. "
            "(2) Position of the European Parliament of 6 April 2022 (not yet published in the "
            "Official Journal) and decision of the Council of 16 May 2022. "
            "(3) Commission Recommendation 2003/361/EC of 6 May 2003 concerning the definition of "
            "micro, small and medium-sized enterprises (OJ L 124, 20.5.2003, p. 36). "
            "The Commission also called for the free and safe flow of data."
        )
        result = _strip_embedded_footnote_blocks(text)
        assert result == (
            "could be pivotal for the rapid development of artificial intelligence technologies. "
            "The Commission also called for the free and safe flow of data."
        )

    def test_handles_long_single_citation_entry(self):
        # Real case from GDPR recital 172: a single footnote entry can
        # itself be very long (a compound title with cross-references to
        # other acts), so the search window for the entry's own end must
        # scale with the inter-marker gap, not a short fixed window.
        text = (
            "to ensure the free movement (1) Regulation (EU) No 536/2014 of the European "
            "Parliament and of the Council of 16 April 2014 on clinical trials on medicinal "
            "products for human use, and repealing Directive 2001/20/EC (OJ L 158, 27.5.2014, "
            "p. 1). "
            "(2) Regulation (EC) No 223/2009 of the European Parliament and of the Council of "
            "11 March 2009 on European statistics and repealing Regulation (EC, Euratom) No "
            "1101/2008 of the European Parliament and of the Council on the transmission of data "
            "subject to statistical confidentiality to the Statistical Office of the European "
            "Communities, Council Regulation (EC) No 322/97 on Community Statistics, and Council "
            "Decision 89/382/EEC, Euratom establishing a Committee on the Statistical Programmes "
            "of the European Communities (OJ L 87, 31.3.2009, p. 164). "
            "of personal data within the Union, the power to adopt acts should be delegated."
        )
        result = _strip_embedded_footnote_blocks(text)
        assert result == (
            "to ensure the free movement of personal data within the Union, "
            "the power to adopt acts should be delegated."
        )

    def test_matches_spaced_footnote_markers(self):
        # Some documents (e.g. the Cybersecurity Act) render footnote
        # numbers with internal spacing: "( 9 )" rather than "(9)".
        text = (
            "was adopted in 2016 in the form of Directive (EU) 2016/1148 of the European "
            "Parliament and of the Council ( 9 ). Directive (EU) 2016/1148 put in place "
            "requirements concerning national capabilities in the field of cybersecurity. "
            "( 6 ) Regulation (EC) No 460/2004 of the European Parliament and of the Council "
            "of 10 March 2004 establishing the European Network and Information Security Agency "
            "(OJ L 77, 13.3.2004, p. 1). "
            "( 7 ) Regulation (EC) No 1007/2008 of the European Parliament and of the Council "
            "of 24 September 2008 amending Regulation (EC) No 460/2004 (OJ L 293, 31.10.2008, "
            "p. 1). "
            "A key role was attributed to ENISA in supporting the implementation of that Directive."
        )
        result = _strip_embedded_footnote_blocks(text)
        assert "( 6 )" not in result
        assert "( 7 )" not in result
        assert "A key role was attributed to ENISA" in result
        # the lone inline "( 9 )" reference is not part of a run and stays
        assert "( 9 )" in result


class TestExtractArticles:
    def test_extracts_multiple_articles_with_headings(self):
        text = (
            "HAVE ADOPTED THIS REGULATION: \n"
            "CHAPTER I \nGeneral provisions \n"
            "Article 1 \nSubject-matter and objectives \n"
            "1. This Regulation lays down rules relating to the protection of natural persons. \n"
            "Article 2 \nMaterial scope \n"
            "1. This Regulation applies to the processing of personal data. \n"
        )
        articles = _extract_articles(text)
        assert len(articles) == 2
        assert articles[0]["number"] == 1
        assert articles[0]["heading"] == "Subject-matter and objectives"
        assert "protection of natural persons" in articles[0]["text"]
        assert articles[1]["number"] == 2
        assert articles[1]["heading"] == "Material scope"

    def test_truncates_at_signature_formula_excluding_annexes(self):
        # Real case from NIS2: without this cutoff, the last article
        # absorbs the signature block and every trailing Annex.
        text = (
            "Article 46 \nAddressees \n"
            "This Directive is addressed to the Member States. "
            "Done at Strasbourg, 14 December 2022. "
            "For the European Parliament The President R. METSOLA "
            "ANNEX I \nSECTORS OF HIGH CRITICALITY \nSector Subsector Type of entity"
        )
        articles = _extract_articles(text)
        assert len(articles) == 1
        assert "ANNEX I" not in articles[0]["text"]
        assert "METSOLA" not in articles[0]["text"]
        assert "addressed to the Member States" in articles[0]["text"]

    def test_no_articles_returns_empty_list(self):
        assert _extract_articles("No enacting terms in this text at all.") == []

    def test_inline_article_reference_is_not_mistaken_for_a_heading(self):
        # "Article N" must be alone on its own line to count as a heading --
        # a reference embedded mid-sentence must not be matched.
        text = (
            "Article 1 \nSubject-matter \n"
            "This Regulation, as provided in Article 2 thereof, applies broadly. \n"
        )
        articles = _extract_articles(text)
        assert len(articles) == 1
        assert articles[0]["number"] == 1
