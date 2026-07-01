"""
templates.py
------------
Complete relation schema for the NLI-RE framework.

This module is the single source of truth for all relation definitions.
It contains three tightly coupled pieces of information for each relation:

  1. RELATION_TEMPLATES  — Arabic verbalization template T_r(e1, e2)
  2. RELATION_SCHEMA     — Allowed domain and range entity types (D, Z)
  3. Helper functions    — verbalize(), build_nli_sentence()

Keeping templates and schema together ensures that adding or modifying
a relation requires changes in only one place.
"""

from typing import Callable, Dict, FrozenSet, List, Set, Tuple

# Type aliases
RelationTemplate = Callable[[str, str], str]
RelationSchema   = Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]]

# ---------------------------------------------------------------------------
# 1. Arabic verbalization templates  (Table 1)
#    Each entry: relation_key -> callable(subject, object) -> hypothesis
# ---------------------------------------------------------------------------

RELATION_TEMPLATES: Dict[str, RelationTemplate] = {
    # ========== Family ===========================================================
    # {e1} هو والد أو والدة {e2}       → is the parent of
    "Family.has_parent":               lambda e1, e2: f"{e1} هو والد أو والدة {e2}",
    # {e1} هو أخ أو أخت {e2}           → is the sibling of
    "Family.has_sibling":              lambda e1, e2: f"{e1} هو أخ أو أخت {e2}",
    # {e1} هو زوج أو زوجة {e2}         → is the spouse of
    "Family.has_spouse":               lambda e1, e2: f"{e1} هو زوج أو زوجة {e2}",
    # {e1} هو قريب {e2}                → is a relative of
    "Family.has_relative":             lambda e1, e2: f"{e1} هو قريب {e2}",

    # ========== Personal =========================================================
    # {e1} وُلِد في تاريخ {e2}          → was born on
    "Personal.birth_date":             lambda e1, e2: f"{e1} وُلِد في تاريخ {e2}",
    # {e1} توفي في تاريخ {e2}           → died on
    "Personal.death_date":             lambda e1, e2: f"{e1} توفي في تاريخ {e2}",
    # {e1} وُلِد في / مكان الولادة {e2} → was born in
    "Personal.birth_place":            lambda e1, e2: f"{e1} وُلِد في / مكان الولادة {e2}",
    # {e1} يعمل كـ / مهنته {e2}        → works as
    "Personal.has_occupation":         lambda e1, e2: f"{e1} يعمل كـ / مهنته {e2}",

    # ========== Business =========================================================
    # {e1} لديه نزاع مع {e2}           → has a conflict with
    "Business.has_conflict_with":      lambda e1, e2: f"{e1} لديه نزاع مع {e2}",
    # {e1} منافس لـ {e2}               → is a competitor of
    "Business.has_competitor":         lambda e1, e2: f"{e1} منافس لـ {e2}",
    # {e1} شريك لـ {e2}                → is a partner of
    "Business.has_partner_with":       lambda e1, e2: f"{e1} شريك لـ {e2}",

    # ========== Administration ===================================================
    # {e1} هو مدير {e2}                → is the manager of
    "Administration.manager_of":       lambda e1, e2: f"{e1} هو مدير {e2}",
    # {e1} هو رئيس/يتولى أعلى منصب في {e2} → is the president of
    "Administration.president_of":     lambda e1, e2: f"{e1} هو رئيس/يتولى أعلى منصب في {e2}",
    # {e1} هو قائد {e2}                → is the leader of
    "Administration.leader_of":        lambda e1, e2: f"{e1} هو قائد {e2}",

    # ========== PartOf ===========================================================
    # {e1} هو تقسيم جغرافي لـ {e2}     → is a geopolitical division of
    "PartOf.geopolitical_division":    lambda e1, e2: f"{e1} هو تقسيم جغرافي لـ {e2}",
    # {e1} فرع تابع لـ {e2}            → is a subsidiary of
    "PartOf.subsidiary":               lambda e1, e2: f"{e1} فرع تابع لـ {e2}",

    # ========== Affiliation ======================================================
    # {e1} عضو في {e2}                 → is a member of
    "Affiliation.member_of":           lambda e1, e2: f"{e1} عضو في {e2}",
    # {e1} يعمل لدى {e2}               → is employed by
    "Affiliation.employee_of":         lambda e1, e2: f"{e1} يعمل لدى {e2}",
    # {e1} طالب في / تلقى تعليمه في {e2} → studies at
    "Affiliation.student_at":          lambda e1, e2: f"{e1} طالب في / تلقى تعليمه في {e2}",
    # {e1} يمتلك {e2}                  → owns
    "Affiliation.owner_of":            lambda e1, e2: f"{e1} يمتلك {e2}",

    # ========== Productivity =====================================================
    # {e1} مخترع {e2}                  → is the inventor of
    "Productivity.inventor_of":        lambda e1, e2: f"{e1} مخترع {e2}",
    # {e1} يصنّع {e2}                  → manufactures
    "Productivity.manufacturer_of":    lambda e1, e2: f"{e1} يصنّع {e2}",
    # {e1} بنى {e2}                    → built
    "Productivity.builder_of":         lambda e1, e2: f"{e1} بنى {e2}",
    # {e1} هو مؤسس {e2}               → is the founder of
    "Productivity.founder_of":         lambda e1, e2: f"{e1} هو مؤسس {e2}",

    # ========== Location =========================================================
    # {e1} يعيش في {e2}               → lives in
    "Location.lives_in":               lambda e1, e2: f"{e1} يعيش في {e2}",
    # {e1} يقع في {e2}                → is located in
    "Location.located_in":             lambda e1, e2: f"{e1} يقع في {e2}",
    # {e1} يقع مقره الرئيسي في {e2}   → is headquartered in
    "Location.headquartered_in":       lambda e1, e2: f"{e1} يقع مقره الرئيسي في {e2}",
    # {e1} لديه حدود مع {e2}          → borders
    "Location.has_border_with":        lambda e1, e2: f"{e1} لديه حدود مع {e2}",
    # {e1} يقع بالقرب من {e2}         → is near
    "Location.nearby":                 lambda e1, e2: f"{e1} يقع بالقرب من {e2}",

    # ========== Organization =====================================================
    # {e1} لديه ممتلكات {e2}           → has property
    "Organization.has_propoerty":      lambda e1, e2: f"{e1} لديه ممتلكات {e2}",
    # {e1} يضم عدد فروع قدره {e2}      → has branches
    "Organization.branch_count":       lambda e1, e2: f"{e1} يضم عدد فروع قدره {e2}",
    # {e1} يحقق إيرادات قدرها {e2}    → generates revenue of
    "Organization.has_revenue":        lambda e1, e2: f"{e1} يحقق إيرادات قدرها {e2}",
    # {e1} عدد موظفيه {e2}             → employs employees
    "Organization.employs":            lambda e1, e2: f"{e1} عدد موظفيه {e2}",
    # {e1} تم تأسيسها بتاريخ {e2}     → was founded on
    "Organization.found_on":           lambda e1, e2: f"{e1} تم تأسيسها بتاريخ {e2}",
    # {e1} يُعرف أيضاً باسم {e2}      → is also known as
    "Organization.has_alternate_name": lambda e1, e2: f"{e1} يُعرف أيضاً باسم {e2}",

    # ========== GPE (Geo-Political Entity) ========================================================
    # {e1} تبلغ مساحتها {e2}           → has an area of
    "GPE.has_area":                    lambda e1, e2: f"{e1} تبلغ مساحتها {e2}",
    # {e1} لغتها الرسمية {e2}          → official language is
    "GPE.official_language":           lambda e1, e2: f"{e1} لغتها الرسمية {e2}",
    # {e1} عملتها هي {e2}              → has currency
    "GPE.has_currency":                lambda e1, e2: f"{e1} عملتها هي {e2}",
    # {e1} هي عاصمة {e2}              → is the capital of
    "GPE.capital_of":                  lambda e1, e2: f"{e1} هي عاصمة {e2}",
    # {e1} عدد سكانها {e2}             → has a population of
    "GPE.has_population":              lambda e1, e2: f"{e1} عدد سكانها {e2}",
}

# ---------------------------------------------------------------------------
# 2. Relation schema: allowed domain (D) and range (Z) entity types (Table 2)
#    Each entry: relation_key -> (domain_frozenset, range_frozenset)
#    Entity type labels match the NER tagset used in the corpus.
# ---------------------------------------------------------------------------

RELATION_SCHEMA: RelationSchema = {
    # ========== Family ===========================================================
    "Family.has_parent":               (frozenset({"PERS"}),                frozenset({"PERS"})),
    "Family.has_spouse":               (frozenset({"PERS"}),                frozenset({"PERS"})),
    "Family.has_sibling":              (frozenset({"PERS"}),                frozenset({"PERS"})),
    "Family.has_relative":             (frozenset({"PERS"}),                frozenset({"PERS"})),

    # ========== Personal =========================================================
    "Personal.birth_date":             (frozenset({"PERS"}),                frozenset({"DATE"})),
    "Personal.death_date":             (frozenset({"PERS"}),                frozenset({"DATE"})),
    "Personal.birth_place":            (frozenset({"PERS"}),                frozenset({"GPE", "LOC"})),
    "Personal.has_occupation":         (frozenset({"PERS"}),                frozenset({"OCC"})),

    # ========== Business =========================================================
    "Business.has_conflict_with":      (frozenset({"ORG", "NORP", "GPE"}),  frozenset({"ORG", "NORP", "GPE"})),
    "Business.has_competitor":         (frozenset({"PERS", "ORG"}),         frozenset({"PERS", "ORG"})),
    "Business.has_partner_with":       (frozenset({"ORG"}),                 frozenset({"ORG"})),

    # ========== Administration ===================================================
    "Administration.manager_of":       (frozenset({"PERS"}),                frozenset({"ORG", "FAC"})),
    "Administration.president_of":     (frozenset({"PERS"}),                frozenset({"ORG", "GPE"})),
    "Administration.leader_of":        (frozenset({"PERS"}),                frozenset({"ORG"})),

    # ========== PartOf ===========================================================
    "PartOf.geopolitical_division":    (frozenset({"GPE", "LOC"}),          frozenset({"GPE", "LOC"})),
    "PartOf.subsidiary":               (frozenset({"ORG"}),                 frozenset({"ORG"})),

    # ========== Affiliation ======================================================
    "Affiliation.member_of":           (frozenset({"PERS", "GPE"}),         frozenset({"ORG", "NORP"})),
    "Affiliation.employee_of":         (frozenset({"PERS"}),                frozenset({"ORG", "FAC"})),
    "Affiliation.student_at":          (frozenset({"PERS"}),                frozenset({"ORG"})),
    "Affiliation.owner_of":            (frozenset({"PERS"}),                frozenset({"ORG", "FAC"})),

    # ========== Productivity =====================================================
    "Productivity.inventor_of":        (frozenset({"PERS"}),                frozenset({"PRODUCT"})),
    "Productivity.manufacturer_of":    (frozenset({"ORG"}),                 frozenset({"PRODUCT"})),
    "Productivity.builder_of":         (frozenset({"PERS", "NORP", "ORG"}), frozenset({"FAC", "ORG"})),
    "Productivity.founder_of":         (frozenset({"PERS"}),                frozenset({"ORG"})),

    # ========== Location =========================================================
    "Location.lives_in":               (frozenset({"PERS", "NORP"}),        frozenset({"GPE", "LOC"})),
    "Location.located_in":             (frozenset({"FAC", "ORG"}),          frozenset({"GPE", "LOC"})),
    "Location.headquartered_in":       (frozenset({"ORG"}),                 frozenset({"LOC", "GPE"})),
    "Location.has_border_with":        (frozenset({"LOC", "GPE"}),          frozenset({"LOC", "GPE"})),
    "Location.nearby":                 (frozenset({"GPE", "LOC", "FAC"}),   frozenset({"GPE", "LOC", "FAC"})),

    # ========== Organization =====================================================
    "Organization.has_propoerty":      (frozenset({"ORG"}),                 frozenset({"PRODUCT"})),
    "Organization.branch_count":       (frozenset({"ORG"}),                 frozenset({"CARDINALITY"})),
    "Organization.has_revenue":        (frozenset({"ORG"}),                 frozenset({"MONEY"})),
    "Organization.employs":            (frozenset({"ORG"}),                 frozenset({"CARDINALITY"})),
    "Organization.found_on":           (frozenset({"ORG"}),                 frozenset({"DATE", "TIME"})),
    "Organization.has_alternate_name": (frozenset({"ORG", "FAC"}),          frozenset({"ORG", "FAC"})),

    # ========== GPE (Geo-Political Entity) ========================================================
    "GPE.has_area":                    (frozenset({"GPE", "LOC"}),          frozenset({"QUANTITY"})),
    "GPE.official_language":           (frozenset({"GPE", "LOC"}),          frozenset({"LANGUAGE"})),
    "GPE.has_currency":                (frozenset({"GPE", "LOC"}),          frozenset({"CURRENCY"})),
    "GPE.has_population":              (frozenset({"GPE"}),                 frozenset({"CARDINALITY"})),
    "GPE.capital_of":                  (frozenset({"GPE"}),                 frozenset({"GPE"})),
}

# ---------------------------------------------------------------------------
# 3. Derived constants
# ---------------------------------------------------------------------------

# All relation types that carry a positive entailment signal
POSITIVE_RELATIONS = set(RELATION_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def verbalize(relation: str, subject: str, obj: str) -> str:
    """
    Verbalize a candidate relation between subject and object into an
    Arabic hypothesis string using the predefined template T_r(e1, e2).

    Args:
        relation: Relation type string (must be a key in RELATION_TEMPLATES).
        subject:  Subject entity mention.
        obj:      Object entity mention.

    Returns:
        Arabic hypothesis string.

    Raises:
        KeyError: If relation is not in RELATION_TEMPLATES.
    """
    if relation not in RELATION_TEMPLATES:
        raise KeyError(
            f"No template defined for relation '{relation}'. "
            f"Available: {sorted(RELATION_TEMPLATES.keys())}"
        )
    return RELATION_TEMPLATES[relation](subject, obj)


def build_nli_sentence(premise: str, hypothesis: str) -> str:
    """
    Construct the full NLI input string:

        [CLS] premise [SEP] hypothesis

    Args:
        premise:    Input sentence containing the entity pair.
        hypothesis: Verbalized relation hypothesis.

    Returns:
        Concatenated string: "[CLS] {premise} [SEP] {hypothesis}"
    """
    return f"[CLS] {premise} [SEP] {hypothesis}"


# ---------------------------------------------------------------------------
# 5. Semantically ambiguous relation groups
#    Relations within the same group are easily confused with one another
#    in natural Arabic discourse because they describe structurally similar
#    relations between the same entity type pairs, differing only in role
#    granularity, formality, or temporal direction.
#
#    Used to generate "hard negatives" for positive records: take a gold
#    (subject, relation, object) triple and substitute relation with a
#    sibling relation from the same group to create a harder, semantically
#    plausible but factually incorrect hypothesis.
# ---------------------------------------------------------------------------

AMBIGUOUS_GROUPS: List[FrozenSet[str]] = [
    # Leadership & Authority — PERS–ORG/GPE authority/affiliation relations
    frozenset({
        "Administration.president_of",
        "Administration.manager_of",
        "Administration.leader_of",
        "Affiliation.employee_of",
    }),
    # Organizational Affiliation — PERS–ORG membership relations
    frozenset({
        "Affiliation.member_of",
        "Affiliation.employee_of",
        "Affiliation.student_at",
    }),
    # Creation & Ownership — PERS/ORG–ORG/FAC/PRODUCT creation or possession
    frozenset({
        "Productivity.founder_of",
        "Affiliation.owner_of",
        "Productivity.builder_of",
        "Productivity.inventor_of",
        "Productivity.manufacturer_of",
    }),
    # Hierarchical Containment — spatial/administrative containment
    frozenset({
        "GPE.capital_of",
        "Location.located_in",
        "PartOf.geopolitical_division",
        "Location.headquartered_in",
    }),
    # Residency & Origin — PERS–GPE/LOC association
    frozenset({
        "Location.lives_in",
        "Personal.birth_place",
    }),
    # Corporate Relations — inter-organizational relations
    frozenset({
        "PartOf.subsidiary",
        "Affiliation.member_of",
        "Business.has_partner_with",
        "Business.has_competitor",
    }),
    # Temporal Life Events — entities linked to life-cycle temporal expressions
    frozenset({
        "Personal.birth_date",
        "Personal.death_date",
    }),
    # Family Relations — kinship relations via similar possessive constructions
    frozenset({
        "Family.has_parent",
        "Family.has_sibling",
        "Family.has_spouse",
        "Family.has_relative",
    }),
    # Proximity & Adjacency — spatial proximity differing in formality
    frozenset({
        "Location.nearby",
        "Location.has_border_with",
    }),
]

# ---------------------------------------------------------------------------
# 6. Relation properties needed for safe ambiguous-negative substitution
# ---------------------------------------------------------------------------

# Symmetric relations: r(a, b) implies r(b, a). For these, swapping subject
# and object yields an equally true statement, so a candidate ambiguous
# relation that is symmetric must also be checked in its swapped form
# before being treated as a safe negative.
SYMMETRIC_RELATIONS: FrozenSet[str] = frozenset({
    "Family.has_sibling",
    "Family.has_spouse",
    "Family.has_relative",
    "Business.has_conflict_with",
    "Business.has_competitor",
    "Business.has_partner_with",
    "Location.has_border_with",
    "Location.nearby",
    "Organization.has_alternate_name",
})

# Transitive containment chain: if a(r)b and b(r)c both hold for relations
# in this chain, then a(r)c can be inferred and must not be used as a
# negative substitution for the (a, c) pair.
TRANSITIVE_CONTAINMENT_RELATIONS: FrozenSet[str] = frozenset({
    "Location.located_in",
    "PartOf.geopolitical_division",
    "PartOf.subsidiary",
})


def get_ambiguous_relations(relation: str) -> Set[str]:
    """
    Return all relations that share an ambiguity group with the given
    relation, excluding the relation itself.

    Args:
        relation: A relation key present in RELATION_TEMPLATES.

    Returns:
        Set of sibling relation keys from the same ambiguity group(s).
        Empty set if the relation does not belong to any group.
    """
    siblings: Set[str] = set()
    for group in AMBIGUOUS_GROUPS:
        if relation in group:
            siblings |= (group - {relation})
    return siblings
