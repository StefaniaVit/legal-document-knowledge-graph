"""
One-off script applying Claude's adjudicated verdicts to the labeling sample,
finalizing data/labeled/entities_review.csv and relations_review.csv.

Every row was reviewed against the source chunk text and this project's own
established entity-type conventions (e.g. "supervisory authority" = ORGANIZATION,
confirmed by the graph hub-entity analysis earlier in this project). For the 32
entity rows Gemini had already judged, this OVERRIDES Gemini's call where it was
self-contradictory (marking a row incorrect while recommending the same type back)
or inconsistent with established project convention -- each such case is noted.

This is AI-assisted review (Claude, with full project context), not independent
human-validated ground truth -- keep that distinction when reporting results.
"""
import csv
from pathlib import Path

LABELED_DIR = Path(__file__).parents[1] / "data" / "labeled"

# (chunk_id, entity_text, entity_type) -> (is_correct, corrected_type, notes)
ENTITY_VERDICTS = {
    ("32016R0679_recital_49_0", "Council Directive 93/13/EEC", "LEGAL_ACT"):
        ("Y", "", "Correct: genuine, specific legal act citation."),
    ("32016R0679_recital_49_0", "(OJ L 95, 21.4.1993, p. 29)", "DATE"):
        ("N", "LEGAL_ACT", "Bare citation fragment of the directive above, wrongly split out as its own DATE entity."),
    ("32016R0679_recital_49_0", "personal data processing", "LEGAL_CONCEPT"):
        ("Y", "", "Gemini marked N but its own corrected_type was identical (LEGAL_CONCEPT) -- self-contradictory. Type and span are both fine."),
    ("32016R0679_recital_13_0", "European Parliament", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32016R0679_recital_13_0", "Council of the European Union", "LEGAL_ACT"):
        ("N", "ORGANIZATION", "The Council is an institution, not a legal act."),
    ("32016R0679_article_47_0", "competent supervisory authority", "ORGANIZATION"):
        ("Y", "", "Gemini downgraded to LEGAL_CONCEPT, but 'supervisory authority' denotes a real institutional body (national DPA) per this project's own established convention (see graph hub-entity analysis) -- overruled."),
    ("32016R0679_article_47_0", "binding corporate rules", "LEGAL_ACT"):
        ("N", "LEGAL_CONCEPT", "A legal mechanism/instrument category, not a specific enacted law."),
    ("32016R0679_article_47_0", "Article 63", "DATE"):
        ("N", "LEGAL_ACT", "An internal article cross-reference, not a date."),
    ("32016R0679_article_43_7", "Commission", "ORGANIZATION"):
        ("Y", "", "Gemini marked N but its own corrected_type was identical (ORGANIZATION) -- self-contradictory. Correct as-is."),
    ("32016R0679_article_43_7", "delegated acts", "OBLIGATION"):
        ("N", "LEGAL_CONCEPT", "A category of legislative instrument, not an obligation."),
    ("32016R0679_article_43_2", "supervisory authority", "ORGANIZATION"):
        ("Y", "", "Same convention as 'competent supervisory authority' above -- overruled Gemini's downgrade."),
    ("32016R0679_article_43_2", "Article 55 or Article 56", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Type is fine (Gemini's own correction matched the original, self-contradictory), but the span improperly merges two distinct article references with 'or' into one entity."),
    ("32019R0881_recital_29_0", "ENISA", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32019R0881_recital_29_0", "CSIRTs network", "LEGAL_ACT"):
        ("N", "ORGANIZATION", "A network of teams is an organization, not a legal act."),
    ("32019R0881_recital_23_0", "ENISA", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32019R0881_recital_23_0", "Member States", "LEGAL_ACT"):
        ("N", "ORGANIZATION", "Clearly an organizational/political entity, not a legal act."),
    ("32019R0881_article_61_0", "European cybersecurity certification scheme", "LEGAL_ACT"):
        ("N", "LEGAL_CONCEPT", "A framework/mechanism, not a specific enacted law."),
    ("32019R0881_article_61_0", "national cybersecurity certification authorities", "ORGANIZATION"):
        ("Y", "", "Gemini's own notes contradicted its corrected_type (both said ORGANIZATION) -- self-contradictory. Institutional-role phrase, correct per project convention."),
    ("32019R0881_article_61_0", "Commission of the European Union", "LEGAL_CONCEPT"):
        ("N", "ORGANIZATION", "Refers to the European Commission, an institution, not a concept."),
    ("32019R0881_article_61_0", "Official Journal of the European Union", "LEGAL_CONCEPT"):
        ("N", "ORGANIZATION", "A formal EU institutional publication, not an abstract concept."),
    ("32019R0881_article_68_0", "Regulation (EU) No 526/2013", "LEGAL_ACT"):
        ("Y", "", "Correct, specific numbered regulation."),
    ("32019R0881_article_68_0", "'ENISA'", "ORGANIZATION"):
        ("Y", "", "Type and core entity correct; stray quote marks are a minor cosmetic artifact, not a misidentification."),
    ("32019R0881_article_68_0", "'Management Board', 'Executive Director, Management Board, Executive Board'", "OBLIGATION"):
        ("N", "ORGANIZATION", "Garbled multi-item span combining several distinct organizational bodies into one string; type is also wrong."),
    ("32019R0881_article_52_0", "cybersecurity certification scheme", "LEGAL_ACT"):
        ("N", "LEGAL_CONCEPT", "A framework/concept, not a specific enacted law."),
    ("32019R0881_article_52_0", "'basic'", "PENALTY"):
        ("N", "LEGAL_CONCEPT", "An assurance-level label, not a penalty."),
    ("32019R0881_article_52_0", "'substantial' or 'high'", "LEGAL_CONCEPT"):
        ("N", "LEGAL_CONCEPT", "Type is defensible but the span improperly merges two distinct assurance-level values into one entity."),
    ("32022L2555_recital_16_0", "Union data protection law", "LEGAL_CONCEPT"):
        ("Y", "", "Correct."),
    ("32022L2555_recital_16_0", "Directive 2002/58/EC of the European Parliament and of the Council (9)", "LEGAL_ACT"):
        ("Y", "", "Correctly identified and typed; trailing '(9)' is a minor footnote-marker artifact, not a misidentification."),
    ("32022L2555_recital_104_0", "public electronic communications networks", "ORGANIZATION"):
        ("N", "LEGAL_CONCEPT", "Refers to a system/infrastructure category, not an organization."),
    ("32022L2555_recital_104_0", "end-to-end encryption as well as data- centric security concepts such as cartography, segmentation, tagging, access policy and access management", "LEGAL_CONCEPT"):
        ("N", "LEGAL_CONCEPT", "Type is fine but the span is a whole clause listing many distinct concepts, not one bounded entity."),
    ("32022L2555_article_18_0", "ENISA", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32022L2555_article_18_0", "European Parliament", "LEGAL_ACT"):
        ("N", "ORGANIZATION", "Clearly an institution, not a legal act."),
    # Rows never reached by Gemini (daily quota exhausted) -- reviewed directly.
    ("32022L2555_article_18_0", "cybersecurity risk assessment", "PENALTY"):
        ("N", "OBLIGATION", "A required assessment/report activity, not a penalty."),
    ("32022L2555_article_18_0", "peer reviews referred to in Article 19", "OBLIGATION"):
        ("Y", "", "A single coherent required activity with its legal basis noted; not multi-item garbling."),
    ("32022L2555_article_2_3", "providers of public electronic communications networks", "ORGANIZATION"):
        ("Y", "", "Institutional-role phrase denoting real organizational actors, per project convention."),
    ("32022L2555_article_2_3", "processing of personal data pursuant to this Directive by providers of publicly available electronic communications services", "LEGAL_CONCEPT"):
        ("N", "LEGAL_CONCEPT", "Type is fine but span is an entire clause, not a bounded entity."),
    ("32022L2555_article_2_3", "Union privacy law, in particular Directive 2002/58/EC", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Improperly merges a general concept ('Union privacy law') with a specific citation into one span."),
    ("32022L2555_article_2_2", "Directive", "LEGAL_CONCEPT"):
        ("N", "LEGAL_ACT", "Bare generic self-reference with no specific identification -- known extraction defect; also a dubious type for a self-reference."),
    ("32022L2555_article_2_2", "(EU) 2016/679", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Truncated citation fragment -- missing 'Regulation', should read 'Regulation (EU) 2016/679'."),
    ("32022L2555_article_2_2", "Regulation (EU)", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Truncated citation fragment -- missing the actual regulation number."),
    ("32022L2555_article_2_2", "Framework Decision", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Generic category reference with no specific identification."),
    ("32022L2555_article_2_2", "Member States", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32022L2555_article_2_2", "(27) Directive 2011/93/EU of the European Parliament and of the Council...", "LEGAL_CONCEPT"):
        ("N", "LEGAL_ACT", "Leading '(27)' is a footnote-marker artifact leaking into the span, and the type is wrong -- this is a specific named directive, not a concept."),
    ("32022R0868_recital_15_0", "anonymous information", "LEGAL_CONCEPT"):
        ("Y", "", "Correct."),
    ("32022R0868_recital_15_0", "personal data rendered anonymous", "LEGAL_CONCEPT"):
        ("Y", "", "Correct."),
    ("32022R0868_recital_15_0", "Regulation (EU) 2016/679", "LEGAL_ACT"):
        ("Y", "", "Correct."),
    ("32022R0868_recital_30_0", "data economy", "LEGAL_CONCEPT"):
        ("Y", "", "Correct."),
    ("32022R0868_recital_30_0", "Union citizens", "ORGANIZATION"):
        ("N", "LEGAL_CONCEPT", "A population/stakeholder category of natural persons, not an organization."),
    ("32022R0868_recital_30_0", "public sector", "LEGAL_ACT"):
        ("N", "LEGAL_CONCEPT", "A general category, not a specific enacted law."),
    ("32022R0868_recital_30_0", "strategic and sensitive data", "PENALTY"):
        ("N", "LEGAL_CONCEPT", "A data category, not a penalty."),
    ("32022R0868_recital_30_0", "Senior Officials Group Mutual Recognition Agreement (MRA)", "LEGAL_CONCEPT"):
        ("N", "LEGAL_ACT", "A specific named agreement is a legal act, not an abstract concept (confirmed during GraphRAG testing earlier in this project)."),
    ("32022R0868_article_11_1", "data intermediation services provider", "ORGANIZATION"):
        ("Y", "", "Institutional-role phrase denoting a real organizational actor."),
    ("32022R0868_article_11_1", "Member States", "LEGAL_CONCEPT"):
        ("N", "ORGANIZATION", "Clearly an organizational/political entity, not a concept."),
    ("32022R0868_article_11_1", "Senior Officials Group Mutual Recognition Agreement (MRA)", "LEGAL_ACT"):
        ("Y", "", "Correct type this time."),
    ("32022R0868_article_35_0", "Commission", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32022R0868_article_35_0", "24 September 2025", "DATE"):
        ("Y", "", "Correct."),
    ("32022R0868_article_35_0", "Regulation", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Bare generic self-reference with no specific identification -- known extraction defect."),
    ("32022R0868_article_2_2", "personal data", "LEGAL_CONCEPT"):
        ("Y", "", "Correct, canonical example."),
    ("32022R0868_article_2_2", "'Article 4, point (1)', of Regulation (EU) 2016/679", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Referent (GDPR Article 4(1)) is real, but span awkwardly merges a sub-provision reference with its parent citation, with stray quote-mark artifacts."),
}

# (chunk_id, subject, relation, object) -> (is_correct, notes)
RELATION_VERDICTS = {
    ("32016R0679_recital_49_0", "personal data processing", "ESTABLISHES", "Council Directive 93/13/EEC"):
        ("N", "Nonsensical -- the directive citation is a stray footnote-bleed artifact in this text, not something 'personal data processing' establishes."),
    ("32016R0679_article_47_0", "binding corporate rules", "IMPOSES_OBLIGATION_ON", "competent supervisory authority"):
        ("N", "Wrong direction: the text says the authority approves the rules, not that the rules impose an obligation on the authority."),
    ("32016R0679_article_43_7", "Commission", "IMPOSES_OBLIGATION_ON", "delegated acts"):
        ("N", "The Commission is empowered to adopt delegated acts -- 'delegated acts' isn't the kind of entity that has obligations imposed on it."),
    ("32019R0881_article_61_0", "European cybersecurity certification scheme", "APPLIES_TO", "national cybersecurity certification authorities"):
        ("Y", "Defensible: the scheme is the framework these authorities operate under."),
    ("32019R0881_article_61_0", "Commission of the European Union", "RESPONSIBLE_FOR", "Official Journal of the European Union"):
        ("N", "Appears fabricated/unsupported -- not a genuine operational responsibility relationship."),
    ("32019R0881_article_68_0", "'ENISA'", "RESPONSIBLE_FOR", "'Management Board', 'Executive Director, Management Board, Executive Board'"):
        ("N", "Object is a garbled multi-item entity (already marked bad in entity review); inherits that defect."),
    ("32022L2555_article_18_0", "ENISA", "APPLIES_TO", "cybersecurity risk assessment"):
        ("N", "Wrong relation type -- ENISA produces/conducts this assessment (closer to RESPONSIBLE_FOR), not something the assessment 'applies to'."),
    ("32022L2555_article_2_2", "(EU) 2016/679", "APPLIES_TO", "Directive"):
        ("N", "Both endpoints are already-broken entities (truncated citation, generic self-reference); relation doesn't parse as meaningful."),
    ("32022L2555_article_2_2", "Regulation (EU)", "IMPOSES_OBLIGATION_ON", "Member States"):
        ("N", "Subject is a truncated citation fragment -- unclear which regulation this refers to."),
    ("32022R0868_article_2_2", "'Article 4, point (1)', of Regulation (EU) 2016/679", "DEFINES", "personal data"):
        ("Y", "The underlying fact is accurate and text-supported (GDPR Art. 4(1) does define personal data), despite the subject's messy span."),
}


def apply_verdicts(path: Path, verdicts: dict, key_fields: list, verdict_fields: list) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    applied = 0
    for row in rows:
        key = tuple(row[f] for f in key_fields)
        if key in verdicts:
            values = verdicts[key]
            for field, value in zip(verdict_fields, values):
                row[field] = value
            applied += 1
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.name}: applied {applied}/{len(rows)} verdicts")


def main() -> None:
    apply_verdicts(
        LABELED_DIR / "entities_review.csv", ENTITY_VERDICTS,
        ["chunk_id", "entity_text", "entity_type"],
        ["is_correct(Y/N)", "corrected_type", "notes"],
    )
    apply_verdicts(
        LABELED_DIR / "relations_review.csv", RELATION_VERDICTS,
        ["chunk_id", "subject", "relation", "object"],
        ["is_correct(Y/N)", "notes"],
    )


if __name__ == "__main__":
    main()
