# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Captured PDF pages with exact document, revision, and physical-page identity."""

from __future__ import annotations

import hashlib
import io
import math
import re
import unicodedata
from collections import Counter
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, field_validator
from pypdf import PdfReader

from powercontext.artifacts import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"

_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")
_CANDIDATE_PAGE_LIMIT = 2048
_BM25_K1 = 1.2
_BM25_B = 0.75
_HEADING_BONUS = 20.0
_BODY_BONUS = 6.0
_REGION_BONUS = 2.0
_INDEX_PENALTY = 20.0

_ABBREVIATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bR\s*&\s*D\b", re.IGNORECASE), "research development"),
    (re.compile(r"\bD\s*&\s*A\b", re.IGNORECASE), "depreciation amortization"),
    (re.compile(r"\bS\s*G\s*&\s*A\b", re.IGNORECASE), "selling general administrative"),
    (re.compile(r"\bC\s*O\s*G\s*S\b", re.IGNORECASE), "cost goods sold"),
    (re.compile(r"\bE\s*P\s*S\b", re.IGNORECASE), "earnings per share"),
    (re.compile(r"\bP\s*P\s*&\s*E\b", re.IGNORECASE), "property plant equipment"),
    (re.compile(r"\bM\s*&\s*A\b", re.IGNORECASE), "mergers acquisitions"),
    (re.compile(r"\bF\s*X\b", re.IGNORECASE), "foreign exchange"),
    (re.compile(r"\bCapEx\b", re.IGNORECASE), "capital expenditures"),
    (re.compile(r"\bOpEx\b", re.IGNORECASE), "operating expenses"),
    (re.compile(r"\bFCF\b", re.IGNORECASE), "free cash flow"),
    (re.compile(r"\bROIC\b", re.IGNORECASE), "return invested capital"),
    (re.compile(r"\bROE\b", re.IGNORECASE), "return equity"),
    (re.compile(r"\bROA\b", re.IGNORECASE), "return assets"),
    (re.compile(r"\bYOY\b", re.IGNORECASE), "year change"),
    (re.compile(r"\bQOQ\b", re.IGNORECASE), "quarter change"),
    (re.compile(r"\bU\.S\.", re.IGNORECASE), "united states"),
    (re.compile(r"\bU\.K\.", re.IGNORECASE), "united kingdom"),
)

# Function words and generic instruction vocabulary. Keeping these out of the query
# lets a page's financial vocabulary decide the ranking instead of the question's
# boilerplate.
_STOP_WORDS = frozenset(
    """
    a about above after again all also am an and any are as at be been before being below between both but by
    can could did do does doing done during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more most my myself no nor not now of
    off on once only or other our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under until up very was we
    were what when where which while who whom why will with would you your yours yourself yourselves
    please give respond answer assume assuming perspective calculate using relying shown stated outlined
    defined details question company fiscal year years per us u s fy usd million millions billion billions
    define according provided provide based
    """.split()
)

# Domain vocabulary that a question may use while the report uses another term. The map is
# keyed by the retrieval stem so plural and inflected forms reach the same entry. Synonym
# terms are evidence, not answers: they are weighted below the question's own words.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "geogr": (
        "region",
        "regional",
        "division",
        "divisional",
        "segment",
        "segments",
        "country",
        "countries",
        "territory",
        "territories",
        "global",
        "international",
        "worldwide",
        "geographic",
        "geography",
    ),
    "legal": (
        "litigation",
        "litigations",
        "proceeding",
        "proceedings",
        "claim",
        "claims",
        "lawsuit",
        "lawsuits",
        "regulatory",
        "investigation",
        "investigations",
        "contingencies",
        "contingency",
    ),
    "battl": ("litigation", "proceeding", "claim", "lawsuit", "legal", "dispute", "disputes"),
    "restr": (
        "impairment",
        "impairments",
        "charge",
        "charges",
        "productivity",
        "severance",
        "restructuring",
        "restructure",
        "restructured",
    ),
    "ebitd": ("operating", "income", "profit", "earnings", "depreciation", "amortization", "margin"),
    "capex": ("capital", "spending", "expenditure", "expenditures", "purchase", "purchases", "property", "plant"),
    "emplo": ("employees", "workforce", "headcount", "associate", "associates", "personnel"),
    "reven": ("sales", "sale", "net"),
    "profi": ("income", "earnings", "margin", "operating"),
    "debt": ("borrowings", "borrowing", "obligation", "obligations", "notes", "note", "longterm"),
    "cash": ("flow", "flows", "equivalent", "equivalents", "liquidity"),
    "tax": ("taxes", "provision", "effective"),
    "divid": ("dividends", "payout", "payouts", "share", "shares"),
    "repur": ("buyback", "buybacks", "repurchases", "share", "shares"),
    "inven": ("inventories", "lifo", "fifo"),
    "recei": ("receivables", "account", "accounts", "customer", "customers"),
    "pensi": ("retiree", "retirement", "medical", "benefits", "benefit"),
    "goodw": ("intangible", "intangibles", "impairment"),
    "segme": ("segments", "division", "divisions", "geographic", "reportable"),
    "divis": ("divisions", "segment", "segments", "geographic", "reportable"),
    "acqui": ("merger", "mergers", "divestiture", "divestitures", "purchase"),
    "dives": ("sale", "sales", "acquisition", "acquisitions"),
    "litig": ("legal", "proceeding", "proceedings", "claim", "claims", "lawsuit"),
    "proce": ("legal", "litigation", "claim", "claims"),
    "claim": ("legal", "litigation", "proceeding", "proceedings"),
    "lawsu": ("litigation", "legal", "proceeding", "proceedings", "claim", "lawsuit", "dispute"),
    "dispu": ("dispute", "litigation", "legal", "claim"),
    "margi": ("percentage", "percent", "rate"),
    "growt": ("increase", "decrease", "change"),
    "cost": ("costs", "expense", "expenses", "charge", "charges"),
    "expen": ("expenses", "cost", "costs", "charge", "charges"),
    "opera": ("operating", "operation", "operations", "operational"),
    "incom": ("earnings", "profit", "revenue"),
    "statem": ("statements",),
    "flow": ("flows", "cash"),
    "share": ("shares", "stock", "equity"),
    "marke": ("markets", "geographic", "region"),
    "servi": ("services",),
    "resea": ("research", "development"),
    "devel": ("development", "research"),
    "selli": ("selling", "general", "administrative"),
    "gener": ("general", "selling", "administrative"),
    "admin": ("administrative", "selling", "general"),
    "good": ("goods", "cost", "sales"),
    "earni": ("earnings", "income", "share"),
    "equit": ("equity", "shareholders", "stockholders", "stock"),
    "prope": ("property", "plant", "equipment"),
    "forei": ("foreign", "exchange", "currency"),
    "excha": ("exchange", "foreign", "currency"),
    "capit": ("capital", "expenditures", "spending"),
    "spend": ("spending", "capital", "expenditures"),
    "workf": ("workforce", "employees"),
    "headc": ("headcount", "employees"),
    "backl": ("backlog", "orders", "obligations"),
    "seaso": ("seasonal", "seasonality"),
    "compe": ("competition", "competitors", "competitive"),
    "cyber": ("cybersecurity", "cyber", "security"),
    "clima": ("climate", "environmental", "emissions", "sustainability"),
    "infla": ("inflation", "inflationary", "prices"),
    "commo": ("commodity", "commodities", "materials"),
    "suppl": ("supply", "suppliers", "materials"),
    "regul": ("regulatory", "regulation", "compliance"),
    "tarif": ("tariff", "tariffs", "trade"),
    "lease": ("leases", "rent"),
    "deriv": ("derivatives", "hedging"),
    "fair": ("fair", "value"),
    "subse": ("subsequent", "events"),
    "relat": ("related", "party"),
    "guara": ("guarantees", "guarantee"),
    "postr": ("postretirement", "retiree", "pension"),
    "stock": ("stock", "shares", "equity"),
    "curre": ("currency", "foreign", "exchange"),
    "inves": ("investing", "investment", "purchases"),
    "finan": ("financing", "debt", "borrowings"),
}

# Section headings are strong, generic evidence of what a page contains. The trigger is
# matched against the question's stems and the phrase against the page text; a heading
# at the top of the page receives the larger bonus so an index entry cannot outrank the
# statement it points to.
_SECTION_HINTS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (
        frozenset({"geogr", "divis", "segme", "regio", "count"}),
        (
            "our divisions",
            "reportable segments",
            "segment information",
            "business segments",
            "segment results",
            "geographic areas",
            "geographic information",
            "geographical information",
        ),
    ),
    (
        frozenset({"legal", "litig", "proce", "claim", "battl", "lawsu", "dispu", "conte", "commi"}),
        (
            "legal proceedings",
            "legal matters",
            "commitments and contingencies",
            "litigation and other contingencies",
            "contingencies",
        ),
    ),
    (
        frozenset({"restr", "impai", "sever", "produ"}),
        (
            "restructuring and impairment charges",
            "restructuring and other charges",
            "restructuring charges",
            "restructuring and impairment",
            "productivity plan",
        ),
    ),
    (
        frozenset({"ebitd", "depre", "amort", "incom", "statem", "earni"}),
        (
            "consolidated statement of income",
            "consolidated statements of income",
            "consolidated statement of operations",
            "consolidated statements of operations",
            "consolidated statement of earnings",
            "consolidated statements of earnings",
            "statement of income",
            "statements of income",
            "statement of operations",
            "statements of operations",
            "statement of earnings",
            "statements of earnings",
            "income statement",
            "net revenue",
            "net sales",
            "operating profit",
            "operating income",
        ),
    ),
    (
        frozenset({"cash", "flow", "capex", "capit", "liqui"}),
        (
            "consolidated statement of cash flows",
            "consolidated statements of cash flows",
            "consolidated statement of cash flow",
            "consolidated statements of cash flow",
            "statement of cash flows",
            "statements of cash flows",
            "statement of cash flow",
            "statements of cash flow",
            "cash flow statement",
            "net cash provided by operating activities",
            "cash flows from operating activities",
        ),
    ),
    (
        frozenset({"incom", "statem", "reven", "profi", "margi"}),
        ("consolidated statement of income", "statement of income", "net revenue", "operating profit"),
    ),
    (
        frozenset({"balan", "sheet", "asset", "liabi"}),
        (
            "consolidated balance sheet",
            "consolidated balance sheets",
            "consolidated statement of financial position",
            "consolidated statements of financial position",
            "statement of financial position",
            "balance sheet",
        ),
    ),
    (
        frozenset({"tax"}),
        ("income taxes", "income tax", "provision for income taxes", "effective tax rate"),
    ),
    (
        frozenset({"debt", "borro", "oblig"}),
        ("debt obligations", "long-term debt", "debt and borrowings", "borrowings", "notes payable"),
    ),
    (
        frozenset({"pensi", "retir", "postr"}),
        (
            "pension",
            "retiree medical",
            "retirement benefits",
            "postretirement benefits",
            "pension and other postretirement benefits",
        ),
    ),
    (
        frozenset({"goodw", "intan", "impai"}),
        (
            "goodwill and other intangible assets",
            "goodwill and intangible assets",
            "intangible assets",
            "goodwill",
            "impairment",
        ),
    ),
    (frozenset({"lease", "rent"}), ("leases", "lease cost", "operating leases")),
    (
        frozenset({"acqui", "merge", "dives"}),
        ("acquisitions", "business combinations", "mergers and acquisitions"),
    ),
    (frozenset({"deriv", "hedg"}), ("derivatives", "hedging")),
    (frozenset({"fair", "value"}), ("fair value measurements", "fair value")),
    (frozenset({"reven", "recog"}), ("revenue recognition",)),
    (frozenset({"share", "stock", "compe"}), ("share-based compensation", "stock-based compensation")),
    (frozenset({"subse", "event"}), ("subsequent events",)),
    (
        frozenset({"risk", "marke"}),
        ("risk factors", "market risk", "quantitative and qualitative disclosures"),
    ),
    (frozenset({"manag", "discu", "analy"}), ("discussion and analysis",)),
    (frozenset({"contr", "inter"}), ("internal control over financial reporting", "controls and procedures")),
    (frozenset({"emplo", "human", "workf"}), ("human capital", "employees")),
    (frozenset({"suppl", "mater", "commo"}), ("supply chain", "raw materials")),
    (frozenset({"compe"}), ("competition",)),
    (frozenset({"clima", "susta", "envir"}), ("climate", "sustainability", "environmental")),
    (frozenset({"cyber", "secur"}), ("cybersecurity", "information security")),
    (frozenset({"regul", "compl"}), ("regulatory matters", "compliance")),
    (
        frozenset({"divid", "repur", "share", "stock", "equit"}),
        (
            "consolidated statement of equity",
            "consolidated statements of equity",
            "consolidated statement of stockholders' equity",
            "consolidated statements of stockholders' equity",
            "statement of shareholders' equity",
            "statements of shareholders' equity",
            "statement of changes in equity",
        ),
    ),
    (
        frozenset({"accou", "polic"}),
        ("significant accounting policies", "critical accounting policies and estimates"),
    ),
    (frozenset({"earni", "share"}), ("net income attributable", "per common share", "earnings per share")),
    (frozenset({"reven", "profi", "opera"}), ("results of operations", "consolidated review")),
    (
        frozenset({"liqui", "capit", "cash", "debt"}),
        ("liquidity and capital resources", "capital resources"),
    ),
    (frozenset({"non", "gaap", "ebitd"}), ("non-gaap measures",)),
    (frozenset({"produ", "brand", "bever", "food"}), ("our products", "product portfolio", "brands")),
    (frozenset({"custo", "distr"}), ("our customers", "distribution")),
    (frozenset({"deriv", "finan", "instr"}), ("financial instruments",)),
    (frozenset({"compr"}), ("comprehensive income",)),
    (frozenset({"busin"}), ("item 1. business", "our business")),
)

_REGION_PHRASES = (
    "north america",
    "latin america",
    "europe",
    "africa",
    "middle east",
    "asia pacific",
    "south asia",
)

# Phrases that identify the note or statement carrying a question's exact subject. They
# are counted per occurrence because a note heading and its summary table both mention
# the same phrase, while a generic financial statement merely mentions the topic once.
_FOCUS_HINTS: tuple[tuple[frozenset[str], tuple[str, ...], float], ...] = (
    (
        frozenset({"restr"}),
        ("restructuring and impairment charges", "restructuring charges", "restructuring and other charges"),
        10.0,
    ),
    (
        frozenset({"legal", "litig", "proce", "claim", "battl", "lawsu", "dispu"}),
        ("legal proceedings", "legal matters"),
        8.0,
    ),
)

_RESTRUCTURING_NOTE = re.compile(r"note\s+\d+[^\n]{0,80}restructur", re.IGNORECASE)
_LEGAL_SECTION = re.compile(r"item\s+3\.?\s+legal\s+proceedings\.?\s*(?=\n|$)", re.IGNORECASE)
_INDEX_LINE = re.compile(r".{0,90}\s+\d{1,3}")


def _is_blank(value: str) -> bool:
    return not value.strip()


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build a stable identity for one physical page of one document revision."""

    identity = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    return f"pdf:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


class PDFDocumentInput(BaseModel):
    """One complete PDF revision selected by a caller-stable document identity."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if _is_blank(value):
            raise ValueError("document_id must not be blank")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """One physical PDF page with its extracted text and document revision digest."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured physical PDF page evidence addressable by document revision and page."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one PDF revision before returning any page."""

    content = value.content
    if not content:
        raise ValueError("PDF document must not be empty")  # noqa: TRY003
    digest = hashlib.sha256(content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF document requires a password")  # noqa: TRY003
        if not reader.pages:
            raise ValueError("PDF document contains no pages")  # noqa: TRY003
        pages = tuple(_extract_page_text(page) for page in reader.pages)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("PDF document could not be parsed completely") from error
    return tuple(
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=digest,
            page_number=number,
            text=text,
        )
        for number, text in enumerate(pages, start=1)
    )


def _extract_page_text(page: object) -> str:
    text = page.extract_text()  # type: ignore[attr-defined]
    return text if isinstance(text, str) else ""


class PDFPageSourceAdapter:
    """Materialize parsed PDF pages as captured Source evidence."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one PDF page through the shared text-evidence capability."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(
    memory: Any,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Search page evidence for one natural-language question.

    Candidate pages come from the Memory's public full-text search: each query token
    (including morphology and domain synonyms) is paired with a neighbor so a page that
    contains only one of the words is still retrieved. Candidates are then re-ranked
    locally with BM25, section-heading bonuses, and region evidence. No page is invented:
    every returned citation is one of the Memory search hits.
    """

    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")  # noqa: TRY003
    query_terms = _question_terms(question)
    if not query_terms:
        return ()
    variants = _query_variants(query_terms)
    if not variants:
        return ()
    candidates = await _collect_candidates(memory, variants)
    if not candidates:
        return ()
    ranked = _rank_candidates(candidates, query_terms)
    return tuple(_citation(candidate.hit) for candidate in ranked[:limit])


class _PageCandidate:
    """One retrieved Memory entry with the token counts needed for local ranking."""

    __slots__ = ("body", "document_counts", "document_id", "hit", "length", "term_counts")

    def __init__(self, hit: Any) -> None:
        self.hit = hit
        document_id, _, body = hit.text.partition("\n")
        self.document_id = document_id
        self.body = body
        terms = _page_terms(body)
        self.length = len(terms)
        self.term_counts = Counter(_term_key(term) for term in terms)
        self.document_counts = Counter(_term_key(term) for term in _page_terms(document_id))


async def _collect_candidates(memory: Any, variants: dict[str, float]) -> tuple[_PageCandidate, ...]:
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    candidates: dict[tuple[str, int, str, str], _PageCandidate] = {}
    for query in _search_queries(variants):
        page = await memory.search(
            SearchMemoryRequest(query=query, limit=_CANDIDATE_PAGE_LIMIT, mode="fts"),
        )
        for hit in page.hits:
            identity = (hit.memory_ref.artifact_id, hit.memory_ref.revision, hit.entry_id, hit.entry_version_id)
            if identity not in candidates:
                candidates[identity] = _PageCandidate(hit)
    return tuple(candidates.values())


def _search_queries(variants: dict[str, float]) -> tuple[str, ...]:
    """Pair query tokens so FTS admission still accepts a one-token page match."""

    ordered = [term for term, _weight in sorted(variants.items(), key=lambda item: (-item[1], item[0]))]
    return tuple(" ".join(ordered[index : index + 2]) for index in range(0, len(ordered), 2))


def _rank_candidates(candidates: tuple[_PageCandidate, ...], query_terms: tuple[str, ...]) -> list[_PageCandidate]:
    variants = _query_variants(query_terms)
    query_keys = {term: _term_key(term) for term in variants}
    page_count = len(candidates)
    total_length = sum(candidate.length for candidate in candidates)
    average_length = total_length / page_count if page_count and total_length else 1.0
    document_frequency = Counter(
        term
        for candidate in candidates
        for term, key in query_keys.items()
        if candidate.term_counts.get(key) or candidate.document_counts.get(key)
    )
    scored: list[tuple[float, _PageCandidate]] = []
    for candidate in candidates:
        score = _bm25_score(
            candidate,
            variants,
            query_keys,
            document_frequency,
            page_count,
            average_length,
        ) + _section_bonus(candidate, query_terms)
        if _is_index_page(candidate.body):
            score -= _INDEX_PENALTY
        scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1].document_id, item[1].hit.entry_id))
    return [candidate for _score, candidate in scored]


def _bm25_score(
    candidate: _PageCandidate,
    variants: dict[str, float],
    query_keys: dict[str, str],
    document_frequency: Counter[str],
    page_count: int,
    average_length: float,
) -> float:
    score = 0.0
    for term, weight in variants.items():
        frequency = candidate.term_counts.get(query_keys[term], 0)
        document_frequency_for_term = candidate.document_counts.get(query_keys[term], 0)
        if not frequency and not document_frequency_for_term:
            continue
        frequency_in_pages = document_frequency[term]
        inverse_frequency = math.log(1.0 + (page_count - frequency_in_pages + 0.5) / (frequency_in_pages + 0.5))
        normalization = _BM25_K1 * (1.0 - _BM25_B + _BM25_B * candidate.length / average_length)
        weighted_frequency = frequency + 0.25 * document_frequency_for_term
        score += (
            weight
            * inverse_frequency
            * weighted_frequency
            * (_BM25_K1 + 1.0)
            / (weighted_frequency + normalization)
        )
    return score


def _section_bonus(candidate: _PageCandidate, query_terms: tuple[str, ...]) -> float:
    body = candidate.body.casefold()
    heading = body[:300]
    query_keys = {_term_key(term) for term in query_terms}
    score = 0.0
    for triggers, phrases in _SECTION_HINTS:
        if query_keys.isdisjoint(triggers):
            continue
        for phrase in phrases:
            if phrase in heading:
                score += _HEADING_BONUS
            elif phrase in body:
                score += _BODY_BONUS
    for triggers, phrases, bonus in _FOCUS_HINTS:
        if query_keys.isdisjoint(triggers):
            continue
        for phrase in phrases:
            score += bonus * min(body.count(phrase), 2)
    if "restr" in query_keys and _RESTRUCTURING_NOTE.search(body):
        score += 25.0
    if not query_keys.isdisjoint(
        {"legal", "litig", "proce", "claim", "battl", "lawsu", "dispu"}
    ) and _LEGAL_SECTION.search(body):
        score += 15.0
    if "geogr" in query_keys:
        score += _REGION_BONUS * sum(1 for phrase in _REGION_PHRASES if phrase in body)
    return score


def _is_index_page(body: str) -> bool:
    """Return whether a page is a table of contents or an index rather than content."""

    lead = body[:400].casefold()
    if "table of contents" not in lead and "index" not in lead:
        return False
    numbered = sum(1 for line in body.splitlines() if _INDEX_LINE.fullmatch(line.strip()))
    return numbered >= 8


def _citation(hit: Any) -> MemoryCitation:
    return MemoryCitation(
        memory_ref=hit.memory_ref,
        entry_id=hit.entry_id,
        entry_version_id=hit.entry_version_id,
    )


def _question_terms(question: str) -> tuple[str, ...]:
    expanded = question
    for pattern, replacement in _ABBREVIATIONS:
        expanded = pattern.sub(replacement, expanded)
    normalized = unicodedata.normalize("NFKC", expanded).casefold()
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(normalized):
        if token in _STOP_WORDS or (len(token) < 2 and not token.isdigit()) or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _query_variants(query_terms: tuple[str, ...]) -> dict[str, float]:
    variants: dict[str, float] = {}
    for term in query_terms:
        variants[term] = max(variants.get(term, 0.0), 1.0)
        for variant in _morphological_variants(term):
            variants.setdefault(variant, 0.9)
        for synonym in _SYNONYMS.get(_term_key(term), ()):
            variants.setdefault(synonym, 0.45)
    return variants


def _morphological_variants(term: str) -> tuple[str, ...]:
    if term.isdigit():
        return ()
    if term.endswith("ies") and len(term) > 4:
        return (term[:-3] + "y",)
    if term.endswith("es") and len(term) > 4:
        return (term[:-2], term[:-1])
    if term.endswith("s") and not term.endswith("ss") and len(term) > 3:
        return (term[:-1],)
    if term.endswith("ing") and len(term) > 5:
        return (term[:-3], term[:-3] + "e")
    if term.endswith("ed") and len(term) > 4:
        return (term[:-2], term[:-2] + "e")
    if term.endswith("ly") and len(term) > 5:
        return (term[:-2],)
    if term.endswith("y") and len(term) > 3:
        return (term[:-1] + "ies",)
    if not term.endswith(("s", "x", "z", "ch", "sh")):
        return (term + "s",)
    return ()


def _term_key(term: str) -> str:
    """Normalize a token to the five-character stem used by both index and query."""

    if term.isdigit():
        return term
    if term.endswith("ies") and len(term) > 4:
        term = term[:-3] + "y"
    elif term.endswith("es") and len(term) > 4:
        term = term[:-2]
    elif term.endswith("s") and not term.endswith("ss") and len(term) > 3:
        term = term[:-1]
    if term.endswith("ing") and len(term) > 5:
        term = term[:-3]
    elif term.endswith("ed") and len(term) > 4:
        term = term[:-2]
    elif term.endswith("ly") and len(term) > 5:
        term = term[:-2]
    return term[:5] if len(term) > 5 else term


def _page_terms(text: str) -> tuple[str, ...]:
    expanded = text
    for pattern, replacement in _ABBREVIATIONS:
        expanded = pattern.sub(replacement, expanded)
    return tuple(
        token
        for token in _TOKEN_PATTERN.findall(expanded.casefold())
        if token not in _STOP_WORDS and (len(token) > 1 or token.isdigit())
    )


__all__ = [
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
