from engines.sponsor_evidence_engine import (
    _classify_item,
    _freshness,
    build_sponsor_evidence_summary,
)
from config.sponsor_evidence_sources import SponsorEvidenceSource


def test_recent_press_release_for_future_conference_is_active():
    state, score, year, reason = _freshness(
        "2026-02-15",
        "Company will present Phase 1 ovarian ADC data at ASCO 2026",
    )
    assert state == "upcoming_catalyst"
    assert year == 2026
    assert score > 0.5
    assert "published" in reason


def test_recent_article_about_old_conference_is_suppressed():
    state, score, year, reason = _freshness(
        "2026-02-15",
        "Company recaps promising preclinical B7-H4 ADC data at AACR 2024",
    )
    assert state == "stale_historical_event"
    assert year == 2024
    assert score < 0.5
    assert "older than current year" in reason


def test_old_press_release_is_suppressed_even_without_conference_year():
    source = SponsorEvidenceSource(
        sponsor="Example Oncology",
        tickers=(),
        aliases=("Example Oncology",),
        evidence_terms=("ovarian", "ADC"),
    )
    item = _classify_item(source, "SCREEN", {
        "title": "Example Oncology will present Phase 1 ovarian ADC data",
        "publisher": "GlobeNewswire",
        "providerPublishTime": "2024-01-15",
        "link": "https://example.com/old-pr",
        "route": "fast_news_screen",
    })
    assert item is not None
    assert item.evidence_state == "stale_historical_event"
    assert item.freshness_state == "stale_publication"


def test_audit_reports_unscreened_runtime_gap(monkeypatch):
    from engines.sponsor_discovery_engine import DiscoveredSponsor
    import engines.sponsor_evidence_engine as see

    monkeypatch.setattr(see, "MAX_FAST_SCREEN_SPONSORS", 2)
    monkeypatch.setattr(see, "MAX_FAST_SCREEN_SECONDS", 10.0)
    monkeypatch.setattr(see, "_news_items_for_ticker", lambda _ticker: [])
    monkeypatch.setattr(see, "_fast_screen_news_items", lambda _source: [])
    monkeypatch.setattr(see, "_ir_newsroom_screen_items", lambda _source: [])

    discovered = []
    for idx in range(5):
        discovered.append(DiscoveredSponsor(
            sponsor_name=f"Example Sponsor {idx}",
            normalized_name=f"Example Sponsor {idx}",
            aliases=(),
            roles=("lead sponsor",),
            matched_lanes=("Ovarian ADC",),
            nct_ids=(f"NCT{idx}",),
            trial_count=1,
            phases=("PHASE1",),
            statuses=("Recruiting",),
            program_terms=("ovarian", "ADC"),
            conditions=("Ovarian Cancer",),
            sponsor_type="Biotech / emerging sponsor",
            last_update="2026-05-01",
            relevance_score=10 - idx,
            evidence_queries=(),
        ))

    summary = build_sponsor_evidence_summary([], discovered_sponsors=discovered)
    assert summary.audit is not None
    assert summary.audit.fast_screen_sponsors == 2
    assert summary.audit.unscreened_sponsors >= 3
    assert summary.audit.unscreened_high_priority
    assert summary.audit.freshness_model == "publication_date_plus_catalyst_timing"


def test_fast_screen_promotes_named_recent_data_signal():
    import engines.sponsor_evidence_engine as see
    source = SponsorEvidenceSource(
        sponsor="Example Biopharma",
        tickers=(),
        aliases=("Example Biopharma",),
        evidence_terms=("ADC", "ovarian"),
    )
    assert see._looks_promising_for_promotion(source, {
        "title": "Example Biopharma Announces Phase 1 Ovarian ADC Data Presentation",
        "publisher": "GlobeNewswire",
        "providerPublishTime": "2026-05-01",
        "link": "https://example.com/press-release",
    })


def test_company_like_sponsors_rank_before_institutional_entities():
    import engines.sponsor_evidence_engine as see
    company = SponsorEvidenceSource("Tubulis", (), ("Tubulis GmbH",), 50, ("ADC", "ovarian"))
    hospital = SponsorEvidenceSource("Universitair Ziekenhuis Brussel", (), (), 50, ("ADC", "ovarian"))
    ranked = see._source_universe([], [], [])
    assert see._source_rank_score(company) < see._source_rank_score(hospital)


def test_generic_ir_newsroom_route_promotes_current_company_release(monkeypatch):
    from engines.sponsor_evidence_engine import build_sponsor_evidence_summary

    def no_ticker_news(_ticker):
        return []

    def no_fast_news(_source):
        return []

    def fake_ir_newsroom(source):
        if source.sponsor == "NextCure":
            return [{
                "title": "NextCure to present SIM0505 Phase 1 dose-escalation data at ASCO 2026",
                "publisher": "NextCure IR/newsroom",
                "providerPublishTime": "2026-04-21",
                "link": "https://ir.example.com/news-releases/sim0505-asco-2026",
                "route": "ir_newsroom_screen",
            }]
        return []

    monkeypatch.setattr("engines.sponsor_evidence_engine._news_items_for_ticker", no_ticker_news)
    monkeypatch.setattr("engines.sponsor_evidence_engine._fast_screen_news_items", no_fast_news)
    monkeypatch.setattr("engines.sponsor_evidence_engine._ir_newsroom_screen_items", fake_ir_newsroom)

    summary = build_sponsor_evidence_summary(["NextCure"])
    assert summary.source_status == "live"
    assert summary.timing_items or summary.result_items
    item = summary.items[0]
    assert item.sponsor == "NextCure"
    assert item.evidence_route == "ir_newsroom_screen"
    assert item.data_stage == "PHASE1"
    assert item.evidence_action == "PLANNED_PRESENTATION"
    assert item.catalyst_year == 2026
    assert item.source_quality == "high"
    assert summary.audit is not None
    assert "generic_ir_newsroom_discovery" in summary.audit.source_routes_checked


def test_company_site_url_detection_not_ir_only():
    from engines.sponsor_evidence_engine import _is_likely_company_evidence_url

    assert _is_likely_company_evidence_url("https://examplebiotech.com/news/phase-1-data")
    assert _is_likely_company_evidence_url("https://examplebiotech.com/media/events/poster-presentation")
    assert _is_likely_company_evidence_url("https://examplebiotech.com/science/publications")
    assert not _is_likely_company_evidence_url("https://clinicaltrials.gov/study/NCT00000000")


def test_private_company_site_screen_is_allowed_in_ranked_pass(monkeypatch):
    from config.sponsor_evidence_sources import SponsorEvidenceSource
    from engines import sponsor_evidence_engine as see

    source = SponsorEvidenceSource("Emergent BioTarget", (), ("Emergent BioTarget Ltd",), 50, ("B7-H4", "ADC", "ovarian"))
    assert see._entity_grade(source) == "sponsor_grade_company"
    assert source.tickers == ()


def test_nextcure_trace_distinguishes_clinical_discovery_from_mapped_evidence(monkeypatch):
    from engines.sponsor_discovery_engine import DiscoveredSponsor
    from engines.sponsor_evidence_engine import build_sponsor_evidence_summary, sponsor_evidence_trace_table
    import engines.sponsor_evidence_engine as see

    monkeypatch.setattr(see, "_news_items_for_ticker", lambda _ticker: [])
    monkeypatch.setattr(see, "_fast_screen_news_items", lambda _source: [])
    monkeypatch.setattr(see, "_discover_company_evidence_candidate_urls", lambda _source: ["https://example.com/news"])
    monkeypatch.setattr(see, "_ir_newsroom_screen_items", lambda _source: [{
        "title": "NextCure to present SIM0505 Phase 1 dose-escalation data at ASCO 2026",
        "publisher": "NextCure newsroom",
        "providerPublishTime": "2026-04-21",
        "link": "https://example.com/news/sim0505-asco-2026",
        "route": "ir_newsroom_screen",
    }])

    discovered = [DiscoveredSponsor(
        sponsor_name="NextCure",
        normalized_name="NextCure",
        aliases=("NextCure, Inc.",),
        roles=("lead sponsor",),
        matched_lanes=("B7-H4 ADC",),
        nct_ids=("NCT06792552",),
        trial_count=1,
        phases=("PHASE1",),
        statuses=("Recruiting",),
        program_terms=("SIM0505", "B7-H4", "ADC"),
        conditions=("Advanced Solid Tumors",),
        sponsor_type="Industry",
        last_update="2026-05-01",
        relevance_score=20,
        evidence_queries=(),
    )]

    summary = build_sponsor_evidence_summary([], discovered_sponsors=discovered)
    assert summary.trace is not None
    assert summary.trace.clinical_discovered is True
    assert "NCT06792552" in summary.trace.clinical_nct_ids
    assert summary.trace.fast_screened is True
    assert summary.trace.company_site_route_attempted is True
    assert summary.trace.promoted_titles
    assert summary.trace.accepted_titles
    trace_df = sponsor_evidence_trace_table(summary)
    assert not trace_df.empty
    assert "ClinicalTrials.gov discovered focus sponsor/program" in set(trace_df["Trace Step"])
