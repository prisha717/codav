from __future__ import annotations
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List
import pandas as pd


@dataclass(frozen=True)
class LinkSuggestion:
    #DTOforonesuggestedlinkpair
    #NoDBlogic
    #JuststructureddataforAPI
    col_a: str
    col_b: str
    confidence: float
    name_score: float
    overlap_score: float
    reason: str


#Smallsynonymmaptohandleinconsistentcolumnnames
#Keeptinyanddefensible
_BASIC_SYNONYMS = {
    "country": {"nation", "country_name", "countryname"},
    "region": {"area"},
    "state": {"province"},
    "date": {"day", "timestamp", "time"},
    "year": {"yr"},
    "code": {"id", "key", "iso"},
    "iso": {"iso2", "iso3"},
}

#Tokensthatcausefalsepositives
#Usedtodownweightgenericmatches
_STOP_TOKENS = {
    "id",
    "key",
    "code",
    "value",
    "values",
    "name",
    "desc",
    "description",
}


def _normalise_text(s: str) -> str:
    #Normalisetextforcolumnnamecomparison
    #Lowercase
    #Replaceunderscoresandhyphenswithspaces
    #Removeotherpunctuation
    #Collapsewhitespace
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_year_col(name: str) -> bool:
    #Detectwideformatyearcolumns
    #Eg1980,1999,2023
    n = _normalise_text(name)
    return bool(re.fullmatch(r"\d{4}", n))


def _is_bad_link_token(name: str) -> bool:
    #BlockcommonnonkeyfieldsforWorldBankstyledata
    n = _normalise_text(name)
    return "indicator" in n


def _tokenise(col: str) -> List[str]:
    #Tokenisecolumnnameintowords
    #Expandwithasmallrulebasedsynonymmap
    #Removeduplicatestokenswhilekeepingorder
    base = _normalise_text(col)
    tokens = [t for t in base.split(" ") if t]

    expanded: List[str] = []
    for t in tokens:
        expanded.append(t)

        #Synonymexpansion
        #Ifatokenmatchesarootoroneofitssynonyms
        #Addtherootanditssynonymset
        for root, syns in _BASIC_SYNONYMS.items():
            if t == root or t in syns:
                expanded.append(root)
                expanded.extend(list(syns))

    seen = set()
    out: List[str] = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    #Jaccardsimilarityontokensets
    #Goodforcaseslikecountry_namevscountryname
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _sequence_ratio(a: str, b: str) -> float:
    #Fuzzystringsimilarity
    #Goodforcaseslikecountrycodevscountry_code
    return SequenceMatcher(None, a, b).ratio()


def _name_similarity(col_a: str, col_b: str) -> float:
    #Combinednamescore
    #Usebothtokenoverlapandfuzzystringratio
    #Takeamaxforforgivingmatching
    #Applypenaltyforgenericidcodeonlymatches
    ta = _tokenise(col_a)
    tb = _tokenise(col_b)

    jac = _jaccard(ta, tb)
    seq = _sequence_ratio(_normalise_text(col_a), _normalise_text(col_b))

    generic_penalty = 0.0
    if (set(ta) <= _STOP_TOKENS) or (set(tb) <= _STOP_TOKENS):
        generic_penalty = 0.2
    score = max(jac, seq) - generic_penalty
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score


def _clean_values(series: pd.Series, max_rows: int) -> List[str]:
    #Preparevaluesforoverlapchecking
    #Dropnulls
    #SamplefirstNrowsforperformance
    #Converttocleanlowercasedstrings
    s = series.dropna()
    if len(s) == 0:
        return []
    s = s.head(max_rows)

    vals: List[str] = []
    for v in s.tolist():
        try:
            txt = str(v).strip().lower()
        except Exception:
            continue
        txt = re.sub(r"\s+", " ", txt)
        if txt:
            vals.append(txt)
    return vals


def _overlap_score(df_a: pd.DataFrame, df_b: pd.DataFrame, col_a: str, col_b: str, max_rows: int) -> float:
    #Optionalconfidenceboostusingvalueoverlap
    #Helpsavoidstupidmatcheslikeidvscode
    #Usesuniquesetsandnormalisesbysmallerset
    if col_a not in df_a.columns or col_b not in df_b.columns:
        return 0.0

    va = _clean_values(df_a[col_a], max_rows=max_rows)
    vb = _clean_values(df_b[col_b], max_rows=max_rows)
    if not va or not vb:
        return 0.0

    sa = set(va)
    sb = set(vb)
    inter = sa & sb

    denom = float(min(len(sa), len(sb)))
    if denom <= 0.0:
        return 0.0

    score = len(inter) / denom
    if score > 1.0:
        score = 1.0
    return score


def shared_columns_exact(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[Dict[str, str]]:
    #Detectexactsharedcolumnsignoringcaseandwhitespace
    #Returnlistofmatchedoriginalnames
    map_a = {_normalise_text(c): c for c in df_a.columns if isinstance(c, str)}
    map_b = {_normalise_text(c): c for c in df_b.columns if isinstance(c, str)}

    shared: List[Dict[str, str]] = []
    for k in sorted(set(map_a.keys()) & set(map_b.keys())):
        if _is_year_col(map_a[k]) or _is_year_col(map_b[k]):
            continue
        shared.append({"col_a": map_a[k], "col_b": map_b[k]})
    return shared


def suggest_links(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    top_k: int = 5,
    name_threshold: float = 0.65,
    max_overlap_rows: int = 2000,
) -> List[LinkSuggestion]:
    #Mainentrypoint
    #LoopoverallcolumnpairsAxB
    #Filterbyname_thresholdtoavoidnoisyresults
    #Filteroutyearonlycolumnsstopwideformatspam
    #Blockindicatorfieldsunlessoverlapexists
    #Blockweakmatcheswithnoevidencefromvalues
    #Computeoverlapboostonasample
    #Combineintosingleconfidence
    #ReturntopKsortedbyconfidence

    suggestions: List[LinkSuggestion] = []

    cols_a = [c for c in df_a.columns if isinstance(c, str)]
    cols_b = [c for c in df_b.columns if isinstance(c, str)]

    for ca in cols_a:
        #SkipyearcolumnsintheAframe
        if _is_year_col(ca):
            continue

        for cb in cols_b:
            #SkipyearcolumnsintheBframe
            if _is_year_col(cb):
                continue

            ns = _name_similarity(ca, cb)
            if ns < name_threshold:
                continue

            os = _overlap_score(df_a, df_b, ca, cb, max_rows=max_overlap_rows)

            #Blockindicatorfieldsunlessactualvalueoverlapexists
            if (_is_bad_link_token(ca) or _is_bad_link_token(cb)) and os == 0.0:
                continue

            #Blockweakmatcheswithnooverlapandlowersimilarity
            #Preventsnoisylinks like CountryName ↔ CountryCode
            if os == 0.0 and ns < 0.85:
                continue

            #Weightedconfidence
            #Nameisprimarysignal
            #Overlapissecondarysanitycheck
            confidence = (0.6 * ns) + (0.4 * os)
            if confidence > 1.0:
                confidence = 1.0

            #ShortexplainablewhystringforUIandviva
            reason_bits = [f"name_similarity={ns:.2f}"]
            if os > 0:
                reason_bits.append(f"value_overlap={os:.2f}")
            reason = ", ".join(reason_bits)

            suggestions.append(
                LinkSuggestion(
                    col_a=ca,
                    col_b=cb,
                    confidence=float(confidence),
                    name_score=float(ns),
                    overlap_score=float(os),
                    reason=reason,
                )
            )

    suggestions.sort(key=lambda x: x.confidence, reverse=True)
    return suggestions[:top_k]


def suggestions_to_dict(sugs: List[LinkSuggestion]) -> List[Dict[str, Any]]:
    #ConvertDTOstojsonfriendlydictsforFastAPIresponses
    #Roundscorestokeepresponsesclean
    out: List[Dict[str, Any]] = []
    for s in sugs:
        out.append(
            {
                "col_a": s.col_a,
                "col_b": s.col_b,
                "confidence": round(s.confidence, 4),
                "name_score": round(s.name_score, 4),
                "overlap_score": round(s.overlap_score, 4),
                "reason": s.reason,
            }
        )
    return out