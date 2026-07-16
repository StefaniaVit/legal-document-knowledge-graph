"""
One-off script applying Claude's adjudicated verdicts to the second
labeling sample (post 7B-upgrade, post chunker-fix), finalizing
data/labeled/entities_review.csv and relations_review.csv.

Unlike the first round (see 06_finalize_labels.py), this one skipped
Gemini pre-review entirely and went straight to direct review, since the
first round showed Gemini's judgments needed heavy correction anyway
(self-contradictions, convention violations) -- this is AI-assisted review
(Claude, with full project context), not independent human-validated
ground truth. Every row was checked against its source chunk text (the
full chunk text, not just the truncated preview in the CSV, for a few
ambiguous cases) and this project's established entity-type conventions.
"""
import csv
from pathlib import Path

LABELED_DIR = Path(__file__).parents[1] / "data" / "labeled"

# (chunk_id, entity_text, entity_type) -> (is_correct, corrected_type, notes)
ENTITY_VERDICTS = {
    ("32016R0679_recital_175_0", "Commission", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32016R0679_recital_175_0", "implementing acts", "LEGAL_ACT"):
        ("N", "LEGAL_CONCEPT", "A generic category of legislative instrument, not a specific enacted law."),
    ("32016R0679_recital_30_0", "controller", "ORGANIZATION"):
        ("Y", "", "Correct, matches established convention."),
    ("32016R0679_recital_30_0", "Regulation", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Bare generic self-reference with no specific identification -- known extraction defect."),
    ("32016R0679_article_4_9", "third party", "LEGAL_CONCEPT"):
        ("Y", "", "Correct, a defined term from a definitions article."),
    ("32016R0679_article_4_9", "natural or legal person, public authority, agency or body", "ORGANIZATION"):
        ("N", "ORGANIZATION", "Garbled multi-item span combining 4 distinct categories into one entity string."),
    ("32016R0679_article_62_1", "supervisory authority", "ORGANIZATION"):
        ("Y", "", "Correct, matches established convention."),
    ("32016R0679_article_62_1", "Member States", "LEGAL_CONCEPT"):
        ("N", "ORGANIZATION", "Clearly an organizational/political entity, not a concept."),
    ("32016R0679_article_28_6", "controller", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32016R0679_article_28_6", "processor", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32016R0679_article_28_6", "standard contractual clauses", "LEGAL_CONCEPT"):
        ("Y", "", "A legal mechanism/instrument category, correctly typed as a concept."),
    ("32019R0881_recital_36_0", "ENISA", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32019R0881_recital_36_0", "Member States and Union institutions, bodies, offices and agencies", "ORGANIZATION"):
        ("N", "ORGANIZATION", "Garbled multi-item span combining several distinct organizational categories into one entity."),
    ("32019R0881_recital_36_0", "cybersecurity risks", "LEGAL_CONCEPT"):
        ("Y", "", "Correct."),
    ("32019R0881_recital_36_0", "emerging technologies", "LEGAL_CONCEPT"):
        ("N", "LEGAL_CONCEPT", "Too generic -- a topic descriptor, not a defined legal/regulatory concept."),
    ("32019R0881_recital_36_0", "technological innovations on network and information security", "LEGAL_CONCEPT"):
        ("N", "LEGAL_CONCEPT", "Span too long (8 words) -- a descriptive clause, not a concise entity."),
    ("32019R0881_recital_32_0", "ENISA", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32019R0881_recital_32_0", "Member States", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32019R0881_recital_32_0", "(EU) 2017/1584", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Truncated citation fragment -- missing the instrument-type prefix (e.g. 'Recommendation')."),
    ("32019R0881_article_7_5", "ENISA", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32019R0881_article_7_5", "Member States", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32019R0881_article_7_5", "(EU) 2016/1148", "LEGAL_ACT"):
        ("N", "LEGAL_ACT", "Truncated citation fragment -- missing the leading 'Directive'."),
    ("32019R0881_article_7_5", "EC3 and CERT-EU", "ORGANIZATION"):
        ("N", "ORGANIZATION", "Garbled span combining two distinct organizations into one entity via 'and'."),
    ("32019R0881_article_5_4", "Union policy", "LEGAL_CONCEPT"):
        ("Y", "", "Correct, a genuine regulatory/policy concept category."),
    ("32019R0881_article_5_4", "European Data Protection Board", "ORGANIZATION"):
        ("Y", "", "Verified present in full chunk text ('providing advice to the European Data Protection Board upon request') -- correct."),
    ("32019R0881_article_63_0", "natural and legal persons", "ORGANIZATION"):
        ("N", "LEGAL_CONCEPT", "A generic population/actor category, not a specific organization (same reasoning as 'Union citizens' in the prior round)."),
    ("32019R0881_article_63_0", "issuer of a European cybersecurity certificate", "LEGAL_ACT"):
        ("N", "ORGANIZATION", "Refers to an actor/role (a conformity assessment body), not a legal act."),
    ("32019R0881_article_63_0", "national cybersecurity certification authority", "LEGAL_CONCEPT"):
        ("N", "ORGANIZATION", "An 'authority' phrase denotes a real institutional body, per established project convention."),
    ("32022L2555_recital_26_0", "sector-specific Union legal act", "LEGAL_ACT"):
        ("N", "LEGAL_CONCEPT", "A generic category description, not a specific named act."),
    ("32022L2555_recital_26_0", "cybersecurity risk-management measures", "OBLIGATION"):
        ("Y", "", "Correct, a real regulatory requirement category."),
    ("32022L2555_recital_26_0", "significant incidents", "LEGAL_CONCEPT"):
        ("Y", "", "Correct."),
    ("32022L2555_recital_124_0", "supervisory authority", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32022L2555_recital_124_0", "Regulation (EU) 2016/679", "LEGAL_ACT"):
        ("Y", "", "Correct, full proper citation (GDPR)."),
    ("32022L2555_article_2_12", "Union", "ORGANIZATION"):
        ("Y", "", "The EU itself as a supranational body, consistent with treating Commission/Council as ORGANIZATION."),
    ("32022L2555_article_2_12", "national rules", "LEGAL_CONCEPT"):
        ("Y", "", "Correct, a generic regulatory category appropriately typed as a concept."),
    ("32022L2555_article_2_12", "business confidentiality", "LEGAL_CONCEPT"):
        ("Y", "", "Correct."),
    ("32022L2555_article_2_12", "Directive 2011/93/EU of the European Parliament and of the Council", "LEGAL_ACT"):
        ("Y", "", "Correct, real specific citation."),
    ("32022L2555_article_2_12", "Council Framework Decision 2004/68/JHA", "LEGAL_ACT"):
        ("Y", "", "Correct, real specific citation."),
    ("32022L2555_article_2_12", "Directive 2013/40/EU of the European Parliament and of the Council", "LEGAL_ACT"):
        ("Y", "", "Correct, real specific citation."),
    ("32022L2555_article_2_10", "Member States", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32022L2555_article_2_10", "obligations", "OBLIGATION"):
        ("N", "OBLIGATION", "Too generic/non-specific to be a meaningful entity -- bare word with no identifying content, analogous to a bare self-reference."),
    ("32022L2555_article_6_14", "vulnerability", "LEGAL_CONCEPT"):
        ("Y", "", "Correct, a defined term from a definitions article."),
    ("32022L2555_article_6_14", "weakness, susceptibility or flaw of ICT products or ICT services that can be exploited by a cyber threat", "LEGAL_CONCEPT"):
        ("N", "LEGAL_CONCEPT", "This is the entire definition clause copied as the entity span, not a bounded entity -- 'vulnerability' above already captures this concept concisely."),
    ("32022R0868_recital_26_0", "Directive 96/9/EC", "LEGAL_ACT"):
        ("Y", "", "Correct, real specific citation (a single inline reference, not part of a footnote-block run, so correctly left in place by the footnote stripper)."),
    ("32022R0868_recital_26_0", "European Parliament and of the Council", "ORGANIZATION"):
        ("N", "ORGANIZATION", "Garbled citation-parsing artifact (fragment of 'Directive 96/9/EC of the European Parliament and of the Council'), not a clean standalone entity."),
    ("32022R0868_recital_29_0", "European Data Innovation Board", "ORGANIZATION"):
        ("Y", "", "A genuine body established by the Data Governance Act -- correct."),
    ("32022R0868_recital_29_0", "Commission", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32022R0868_article_14_2", "competent authority for data intermediation services", "ORGANIZATION"):
        ("Y", "", "Correct, matches 'authority' = ORGANIZATION convention."),
    ("32022R0868_article_14_2", "data intermediation services provider", "LEGAL_CONCEPT"):
        ("N", "ORGANIZATION", "Refers to a real organizational actor (a provider/company) -- matches this same entity's correct classification in the prior labeling round."),
    ("32022R0868_article_19_4", "competent authority for the registration of data altruism organisations", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32022R0868_article_19_4", "Commission", "ORGANIZATION"):
        ("Y", "", "Correct."),
    ("32022R0868_article_2_1", "natural or legal persons", "ORGANIZATION"):
        ("N", "LEGAL_CONCEPT", "A generic population/actor category, not a specific organization."),
    ("32022R0868_article_2_1", "public sector bodies", "LEGAL_CONCEPT"):
        ("N", "ORGANIZATION", "Refers to real government/organizational bodies, not an abstract concept."),
}

# (chunk_id, subject, relation, object) -> (is_correct, notes)
RELATION_VERDICTS = {
    ("32016R0679_recital_175_0", "Commission", "IMPOSES_OBLIGATION_ON", "implementing acts"):
        ("N", "Full text says the Commission 'should adopt' implementing acts -- a real connection, but "
              "IMPOSES_OBLIGATION_ON is the wrong relation type for 'adopts' (ESTABLISHES would fit "
              "better); object entity was also independently marked wrong type."),
    ("32019R0881_article_63_0", "natural and legal persons", "GRANTS_RIGHT_TO", "issuer of a European cybersecurity certificate"):
        ("N", "Backwards: the text says persons themselves HAVE the right to lodge a complaint WITH the "
              "issuer -- persons aren't granting a right TO the issuer. Both entities were also "
              "independently marked wrong."),
    ("32022L2555_article_2_10", "obligations", "SUBJECT_TO", "Member States"):
        ("N", "Text says obligations don't require disclosure contrary to Member States' security interests "
              "-- a limitation, not a 'subject to' relationship. Subject entity was also marked too generic."),
    ("32022L2555_article_6_14", "vulnerability", "DEFINES", "weakness, susceptibility or flaw of ICT products or ICT services that can be exploited by a cyber threat"):
        ("N", "Direction/framing is awkward for a 'X means Y' definitions-article pattern, and the object "
              "is the same over-long clause already flagged as a bad entity span."),
    ("32022R0868_article_19_4", "competent authority for the registration of data altruism organisations", "RESPONSIBLE_FOR", "Commission"):
        ("N", "Full text says the authority 'shall notify the Commission' -- a reporting relationship, not "
              "the authority being responsible FOR the Commission (backwards institutional direction)."),
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
