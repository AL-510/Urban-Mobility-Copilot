"""Explanation engine: generates grounded natural-language explanations for route recommendations."""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExplanationEngine:
    """Generates human-readable explanations for route recommendations.

    Uses template-based generation grounded in:
    1. Model prediction outputs (disruption prob, delay, uncertainty)
    2. Retrieved evidence (advisories, alerts)
    3. Route comparison data
    """

    def explain_recommendation(
        self,
        recommended: dict,
        alternatives: list[dict],
        evidence: list[dict] | None = None,
        weather: dict | None = None,
    ) -> dict:
        """Generate explanation for why a route is recommended.

        Returns:
            Dict with:
            - summary: one-sentence recommendation
            - reasoning: paragraph explaining the decision
            - factors: list of contributing factors
            - evidence_citations: references to retrieved documents
            - comparison: brief comparison to alternatives
        """
        rec = recommended
        name = rec["name"]
        time_min = rec["predicted_time_s"] / 60
        disruption = rec["disruption_prob"]
        reliability = rec["reliability_score"]
        delay_median = rec.get("total_delay_median_min", 0)

        # Summary
        summary = self._build_summary(rec, alternatives)

        # Reasoning
        reasoning_parts = []
        q90_min = rec['predicted_time_q90_s'] / 60
        uncertainty_range = q90_min - time_min
        reasoning_parts.append(
            f"{name} is recommended with an estimated travel time of {time_min:.0f} minutes "
            f"(worst-case {q90_min:.0f} minutes, uncertainty range: {uncertainty_range:.0f} min)."
        )

        if disruption < 0.2:
            reasoning_parts.append(
                "This route has low disruption risk, making it a reliable choice."
            )
        elif disruption < 0.5:
            reasoning_parts.append(
                f"While there is moderate disruption risk ({disruption:.0%}), "
                f"expected delays are manageable at around {delay_median:.0f} minutes."
            )
        else:
            reasoning_parts.append(
                f"Despite elevated disruption risk ({disruption:.0%}), "
                "this route still offers the best balance among available options."
            )

        if reliability > 0.8:
            reasoning_parts.append("Travel time predictions are highly consistent, suggesting reliable conditions.")
        elif reliability < 0.5:
            reasoning_parts.append("Note: there is significant uncertainty in travel time estimates. Consider allowing extra buffer time.")

        # Evidence grounding
        evidence_citations = []
        if evidence:
            relevant = self._filter_relevant_evidence(evidence, rec)
            for ev in relevant[:3]:
                citation = {
                    "doc_id": ev.get("doc_id", ""),
                    "title": ev.get("title", ""),
                    "snippet": ev.get("body", "")[:200],
                    "relevance": ev.get("relevance_score", 0),
                    "source": ev.get("source", ""),
                }
                evidence_citations.append(citation)

                if ev.get("incident_type") == "weather":
                    reasoning_parts.append(
                        f"Weather conditions are a factor: {ev.get('title', 'adverse weather')}."
                    )
                elif ev.get("severity") in ("high", "critical"):
                    reasoning_parts.append(
                        f"Active alert: {ev.get('title', 'disruption reported')}. "
                        "The recommended route minimizes exposure to this disruption."
                    )

        # Weather context
        if weather:
            severity = weather.get("weather_severity", 0)
            if severity > 0.5:
                reasoning_parts.append(
                    "Current weather conditions may affect travel. "
                    "The recommended route accounts for weather-related delays."
                )

        # Comparison to alternatives
        comparison_parts = []
        for alt in alternatives[:2]:
            if alt.get("is_recommended"):
                continue
            alt_time = alt["predicted_time_s"] / 60
            alt_risk = alt["disruption_prob"]
            delta_time = alt_time - time_min

            if delta_time > 5:
                comparison_parts.append(
                    f"{alt['name']} would take ~{delta_time:.0f} minutes longer."
                )
            if alt_risk > disruption + 0.15:
                comparison_parts.append(
                    f"{alt['name']} has higher disruption risk ({alt_risk:.0%} vs {disruption:.0%})."
                )
            elif alt_risk < disruption - 0.15:
                comparison_parts.append(
                    f"{alt['name']} has lower risk but is slower."
                )

        # Contributing factors
        factors = []
        for rf in rec.get("risk_factors", []):
            factors.append({
                "type": rf["type"],
                "severity": rf["severity"],
                "description": rf["description"],
                "source": "model_prediction",
            })

        if evidence_citations:
            for cit in evidence_citations:
                factors.append({
                    "type": "evidence",
                    "severity": "info",
                    "description": cit["title"],
                    "source": "retrieved_advisory",
                })

        return {
            "summary": summary,
            "reasoning": " ".join(reasoning_parts),
            "factors": factors,
            "evidence_citations": evidence_citations,
            "comparison": " ".join(comparison_parts) if comparison_parts else "No significantly different alternatives available.",
            "confidence": self._confidence_level(reliability, disruption),
        }

    def explain_rejection(self, route: dict, recommended: dict) -> str:
        """Explain why a specific route was not recommended."""
        rec_time = recommended["predicted_time_s"] / 60
        route_time = route["predicted_time_s"] / 60
        rec_risk = recommended["disruption_prob"]
        route_risk = route["disruption_prob"]

        parts = [f"{route['name']} was not recommended because:"]

        if route_time > rec_time * 1.15:
            parts.append(f"It is approximately {route_time - rec_time:.0f} minutes slower.")

        if route_risk > rec_risk + 0.1:
            parts.append(f"It has higher disruption risk ({route_risk:.0%} vs {rec_risk:.0%}).")

        if route.get("reliability_score", 0) < recommended.get("reliability_score", 0) - 0.1:
            parts.append("It has less predictable travel times.")

        if len(parts) == 1:
            parts.append("The recommended route offers a better overall balance of speed, reliability, and risk.")

        return " ".join(parts)

    def _build_summary(self, rec: dict, alternatives: list[dict]) -> str:
        time_min = rec["predicted_time_s"] / 60
        modes = rec.get("modes", ["drive"])
        mode_str = " + ".join(modes)

        if rec["disruption_prob"] < 0.2:
            return (
                f"Recommended: {rec['name']} ({mode_str}) — "
                f"{time_min:.0f} min with low disruption risk."
            )
        else:
            return (
                f"Recommended: {rec['name']} ({mode_str}) — "
                f"{time_min:.0f} min, best option given current conditions."
            )

    def _filter_relevant_evidence(self, evidence: list[dict], route: dict) -> list[dict]:
        """Filter evidence documents relevant to this route."""
        coords = route.get("coordinates", [])
        if not coords:
            return evidence[:3]

        relevant = []
        for ev in evidence:
            ev_lat = ev.get("lat", 0)
            ev_lon = ev.get("lon", 0)

            # Check if evidence is near any route coordinate
            for coord in coords:
                dist = ((coord[0] - ev_lat) ** 2 + (coord[1] - ev_lon) ** 2) ** 0.5
                if dist < 0.01:  # ~1km
                    relevant.append(ev)
                    break

        # Sort by relevance score if available
        relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant

    def _confidence_level(self, reliability: float, disruption: float) -> str:
        if reliability > 0.8 and disruption < 0.2:
            return "high"
        elif reliability > 0.5 and disruption < 0.5:
            return "medium"
        else:
            return "low"
