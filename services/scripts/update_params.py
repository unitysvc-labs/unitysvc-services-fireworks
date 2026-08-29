#!/usr/bin/env python3
"""
Template-based update_services.py for Fireworks.

This script yields model dictionaries that are rendered using Jinja2 templates.
Much simpler than the DataBuilder approach - just yield dicts with the data.

Usage: python scripts/update_services_template.py
"""

import os
from decimal import Decimal, ROUND_HALF_UP
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

# Will be provided by unitysvc_sellers
from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator


def _as_positive_int(value) -> int | None:
    """Coerce ``value`` to a positive ``int`` or ``None``.

    Fireworks occasionally ships numeric metadata as strings (``"0"``,
    ``"743911218432"``) or as zero-valued placeholders for non-LLM
    services (image-gen models report ``contextLength: 0``).  The platform
    validator wants a strict positive int or null, so anything that isn't
    a parseable, strictly-positive integer becomes ``None``.
    """
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None



# What the platform adds on top of the upstream rate for the MANAGED channel,
# where UnitySVC's key pays Fireworks and the customer pays UnitySVC. The byok
# channel is unaffected: the customer's own key pays Fireworks directly, so
# there is nothing to mark up and nothing to pay out.
#
# The seller's own list price, applied here at populate time rather than by the
# platform, so the stored rate is exactly the rate billed.
PLATFORM_MARKUP = Decimal("1.15")

# 3dp keeps the effective markup within ~15.0-15.7% across the rate range here;
# 2dp would swing a cheap model as wide as 25%.
PRICE_PLACES = Decimal("0.001")

# Every money key a scraped pricing dict may carry. `price` covers the unified
# and per-image shapes; the rest are the token shapes.
_RATE_KEYS = ("input", "cached_input", "output", "price")


def _mk(value: str) -> str:
    """One upstream rate, marked up and rounded, trailing zeros dropped."""
    marked = (Decimal(str(value)) * PLATFORM_MARKUP).quantize(
        PRICE_PLACES, rounding=ROUND_HALF_UP
    )
    return format(marked.normalize(), "f")


def _marked_up(pricing: dict | None) -> dict | None:
    """The customer-facing twin of a scraped upstream pricing dict.

    Marks up every rate key present and rewrites the description to the
    marketplace grammar. `reference` is dropped: it points at Fireworks' own
    pricing page, which quotes the upstream rate, and citing it beside a marked-
    up number would be wrong.
    """
    if not pricing:
        return None
    out = {k: v for k, v in pricing.items() if k not in _RATE_KEYS + ("description", "reference")}
    for k in _RATE_KEYS:
        if pricing.get(k):
            out[k] = _mk(pricing[k])
    if out.get("input") and out.get("output"):
        rates = "$" + out["input"] + "/$" + out["output"]
        unit = "1M input/output tokens"
        if out.get("cached_input"):
            rates = "$" + out["input"] + "/$" + out["cached_input"] + "/$" + out["output"]
            # "1M in/cached/out tokens" is 23 chars — under the parser's 24-char
            # unit cap, where "1M input/cached input/output tokens" is not.
            unit = "1M in/cached/out tokens"
        out["description"] = rates + " / " + unit
    elif out.get("price"):
        out["description"] = "$" + out["price"] + (
            " / image" if pricing.get("type") == "image" else " / 1M tokens"
        )
    return out


class FireworksModelSource:
    """Fetches model data from Fireworks.ai API and yields template dictionaries."""

    # Fields to extract from API response into details.
    # Note: ``context_length`` is the canonical (snake_case) name required by
    # the platform validator (unitysvc#863).  Fireworks emits ``contextLength``
    # in camelCase; ``_extract_details`` translates the legacy key, and we keep
    # the canonical name in the constant so the destination key is obvious.
    TOP_LEVEL_DETAIL_FIELDS = [
        "calibrated",
        "cluster",
        "context_length",
        "defaultDraftModel",
        "defaultDraftTokenCount",
        "defaultSamplingParams",
        "deprecationDate",
        "fineTuningJob",
        "githubUrl",
        "huggingFaceUrl",
        "importedFrom",
        "kind",
        "rlTunable",
        "snapshotType",
        "supportedPrecisions",
        "supportedPrecisionsWithCalibration",
        "supportsImageInput",
        "supportsLora",
        "supportsMtp",
        "supportsTools",
        "teftDetails",
        "trainingContextLength",
        "tunable",
        "useHfApplyChatTemplate",
    ]

    # Same canonical-rename note as TOP_LEVEL_DETAIL_FIELDS:
    # ``parameter_count`` replaces upstream ``parameterCount``.
    BASE_MODEL_DETAIL_FIELDS = [
        "checkpointFormat",
        "defaultPrecision",
        "modelType",
        "moe",
        "parameter_count",
        "supportsFireattention",
        "worldSize",
    ]

    # Upstream camelCase → canonical snake_case rename map applied in
    # ``_extract_details``.  Falls back gracefully if Fireworks ever sends the
    # canonical name itself.
    FIELD_RENAMES = {
        "contextLength": "context_length",
        "parameterCount": "parameter_count",
    }

    def __init__(self, api_key: str, api_base_url: str, model_base_url: str):
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.model_base_url = model_base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        })
        self.fetcher = ModelDataFetcher()
        # Models dropped because the pricing scrape came back empty.  Counted,
        # not just printed, because it decides whether this run may deprecate:
        # the rates are scraped out of a marketing page, so a timeout, a 404
        # or a layout change all look exactly like a model that was retired.
        self.skipped_no_pricing = 0

    def close(self) -> None:
        self.fetcher.close()
        self.session.close()

    def iter_models(self) -> Iterator[dict]:
        """
        Yield model dictionaries for template rendering.

        Each dict contains all variables needed by offering.json.j2 and listing.json.j2.
        """
        models = self._fetch_all_models()
        if not models:
            # Exit non-zero.  An empty enumeration means the upstream call
            # failed, and yielding nothing writes no params, opens no PR and
            # is indistinguishable from "no changes today" — while telling the
            # writer that every committed service has been retired.
            print("Error: Fireworks returned an empty model list")
            sys.exit(1)
        total = len(models)

        for i, model in enumerate(models, 1):
            model_name = model.get("name", "")
            if not model_name:
                continue

            short_name = model_name.split("/")[-1]
            print(f"[{i}/{total}] {short_name}")

            # Keep only models on the serverless menu. ``supportsServerless`` is
            # the stable "callable via serverless inference" signal;
            # ``deployedModelRefs`` tracks live deployment instances, which is too
            # narrow/volatile (a serverless model reads 0 whenever no instance is
            # warm) and dropped valid models. Filter from the list record before
            # the per-model detail fetch so we don't detail-fetch all ~300.
            if not model.get("supportsServerless"):
                print("  Skipped: not serverless")
                continue

            # Get detailed model info
            model_data = self._get_model_details(model_name) or {}

            # Extract pricing from web page
            pricing = self._extract_pricing(short_name)
            if not pricing:
                print("  Skipped: No pricing found")
                self.skipped_no_pricing += 1
                continue

            # Determine service type
            service_type = self._determine_service_type(model_name, pricing)
            is_flux = "flux" in model_name.lower()

            # Build details dict from API response, then backfill missing
            # canonical metadata (context_length / parameter_count) from the
            # OpenRouter → LiteLLM → HuggingFace fallback chain.
            details = self._extract_details(model_data)
            if service_type == "llm":
                self._apply_canonical_fallback(details, model_name)

            # Derive capabilities from service type and model details
            capabilities = self._derive_capabilities(service_type, model_data, model_name)

            # Yield the template variables
            yield {
                # The service's name, which is also its path under specs/ ==
                # listing.name == "fireworks/<short>" (flat layout, #1263).
                # Required by write_params_from_iterator since unitysvc-sellers
                # 0.3.1; there is no `name_field` fallback any more.
                "service_name": f"fireworks/{short_name}",
                "provider_name": "fireworks",

                # Bare model name — offering.name + listing.display_name.
                # Named "model" (not "name") so it survives param extraction:
                # the writer strips name/service_name/provider_name from the
                # stored parameters and re-derives them from the file's path.
                "model": short_name,

                # Routing-key model — Fireworks expects the fully-qualified
                # path ("accounts/fireworks/models/<short>") in the API's
                # ``model`` field, not the short name.  Kept as a separate
                # key (matching the ``offering_name`` convention used by
                # deepseek / cerebras / groq / anthropic templates) so the
                # template doesn't have to know how each provider names
                # things.
                "offering_name": model_name,

                # Offering fields
                "display_name": model_data.get("displayName"),
                "description": model_data.get("description", ""),
                "service_type": service_type,
                "capabilities": capabilities,
                "status": model_data.get("state", "READY").lower(),
                "model_name": model_name,
                "details": details,

                # Listing fields
                "listing_status": "ready" if model_data.get("state") == "READY" else "draft",
                # UPSTREAM rates, exactly as scraped — this is what Fireworks
                # charges and therefore what the platform owes the seller on the
                # managed channel.
                "pricing": pricing,
                # The same rates plus the platform markup: what the CUSTOMER pays
                # on the managed channel. Kept as a separate key so the scraped
                # truth is never overwritten and the two can be compared.
                "managed_pricing": _marked_up(pricing),
                "is_flux": is_flux,
            }

            print(f"  OK: {service_type}")
            time.sleep(0.3)  # Rate limiting

    def _fetch_all_models(self) -> list[dict]:
        """Fetch all models from Fireworks.ai API with pagination."""
        print("Fetching models from Fireworks.ai API...")
        url = f"{self.api_base_url}/accounts/fireworks/models"
        all_models = []
        page_token = None

        while True:
            params = {"pageSize": 200}
            if page_token:
                params["pageToken"] = page_token

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            page_models = data.get("models", [])
            if not page_models:
                break

            all_models.extend(page_models)
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        print(f"Found {len(all_models)} models\n")
        return sorted(all_models, key=lambda x: x["name"])

    def _get_model_details(self, model_name: str) -> dict | None:
        """Get detailed model information from API."""
        endpoint = f"{self.api_base_url}/{model_name}"
        try:
            response = self.session.get(endpoint, timeout=10)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass
        return None

    def _extract_pricing(self, model_slug: str) -> dict | None:
        """Extract pricing from model's web page."""
        clean_slug = model_slug.replace("/", "-").replace(":", "-").lower()
        pricing_url = f"{self.model_base_url}/{clean_slug}"

        try:
            response = self.session.get(pricing_url, timeout=10)
            if response.status_code == 404:
                return None
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            container = soup.find("h3", string=re.compile(r"Available Serverless", re.IGNORECASE))

            if container:
                parent = container.find_parent("div")
                if parent:
                    parent = parent.find_parent("div")
                if parent:
                    text = parent.get_text()

                    # Input/cached_input/output pricing (new format)
                    match = re.search(
                        r"\$\s*(\d+\.?\d*)\s*/\s*\$\s*(\d+\.?\d*)\s*/\s*\$\s*(\d+\.?\d*)\s*Per\s*\d+[MKmk]?\s*Tokens",
                        text, re.IGNORECASE
                    )
                    if match:
                        return {
                            "type": "one_million_tokens",
                            "input": match.group(1),
                            "cached_input": match.group(2),
                            "output": match.group(3),
                            "description": f"Pricing Per 1M Tokens (input/cached input/output): ${match.group(1)} / ${match.group(2)} / ${match.group(3)}",
                            "reference": pricing_url,
                        }

                    # Input/output pricing (legacy format)
                    match = re.search(
                        r"\$\s*(\d+\.?\d*)\s*/\s*\$\s*(\d+\.?\d*)\s*Per\s*\d+[MKmk]?\s*Tokens",
                        text, re.IGNORECASE
                    )
                    if match:
                        return {
                            "type": "one_million_tokens",
                            "input": match.group(1),
                            "output": match.group(2),
                            "description": f"Pricing Per 1M Tokens (input/output): ${match.group(1)} / ${match.group(2)}",
                            "reference": pricing_url,
                        }

                    # Unified token pricing
                    match = re.search(
                        r"\$\s*(\d+\.?\d*)\s*Per\s*\d+[MKmk]?\s*Tokens",
                        text, re.IGNORECASE
                    )
                    if match:
                        return {
                            "type": "one_million_tokens",
                            "price": match.group(1),
                            "description": f"Pricing Per 1M Tokens: ${match.group(1)}",
                            "reference": pricing_url,
                        }

                    # Image pricing
                    match = re.search(r"\$\s*(\d+\.?\d*)\s*Per\s*Image", text, re.IGNORECASE)
                    if match:
                        return {
                            "type": "image",
                            "price": match.group(1),
                            "description": f"Pricing Per Image: ${match.group(1)}",
                            "reference": pricing_url,
                        }

                    # Step pricing
                    match = re.search(r"\$\s*(\d+\.?\d*)\s*Per\s*Step", text, re.IGNORECASE)
                    if match:
                        return {
                            "type": "step",
                            "price": match.group(1),
                            "description": f"Pricing Per Step: ${match.group(1)}",
                            "reference": pricing_url,
                        }

        except requests.RequestException:
            pass

        return None

    def _determine_service_type(self, model_name: str, pricing: dict | None) -> str:
        """Determine service type from model name and pricing."""
        model_lower = model_name.lower()

        if pricing and pricing.get("type") in ("image", "step"):
            return "image_generation"

        if any(kw in model_lower for kw in ["embedding", "embed"]):
            return "embedding"
        if any(kw in model_lower for kw in ["flux", "dalle", "stable-diffusion"]):
            return "image_generation"
        if any(kw in model_lower for kw in ["whisper", "audio", "speech"]):
            return "prerecorded_transcription"

        return "llm"

    #: ``service_type`` -> platform capability vocabulary
    #: (unitysvc ``docs/capabilities.yml``).
    _SERVICE_TYPE_CAPABILITY = {
        "llm": "chat",
        "image_generation": "image-generate",
        "embedding": "embed",
        "prerecorded_transcription": "speech-to-text",
    }

    def _derive_capabilities(self, service_type: str, model_data: dict,
                             model_name: str = "") -> list[str]:
        """The platform capability this offering provides.

        One entry, from the platform vocabulary (unitysvc
        ``docs/capabilities.yml``): a capability names what the caller GETS.

        ``supportsTools`` / ``supportsImageInput`` are deliberately NOT
        folded in.  They are attributes: they change what may appear in a
        chat request, not what comes back.  Both a tool-using model and a
        plain one answer with generated text, so both are ``chat``.  The
        flags stay in ``details`` where Fireworks put them.

        Rerankers are special-cased on the model name because neither
        upstream signal identifies them: ``_determine_service_type`` keys on
        "embed"/"embedding" and so files them as ``llm``, and Fireworks' own
        ``kind`` reports ``EMBEDDING_MODEL``.  A reranker returns reordered
        documents, not a vector and not generated text.
        """
        if "rerank" in model_name.lower():
            return ["rerank"]
        return [self._SERVICE_TYPE_CAPABILITY.get(service_type, "chat")]

    def _extract_details(self, model_data: dict) -> dict:
        """Extract details dict from API response.

        Translates Fireworks' camelCase ``contextLength`` / ``parameterCount``
        to the canonical snake_case names required by the platform validator
        (unitysvc#863).  Both keys are guaranteed to be present in the result
        (defaulting to ``None``) so callers can rely on key presence.
        ``context_length`` / ``parameter_count`` are coerced to a positive
        ``int`` or ``None`` — Fireworks sometimes ships them as strings, and
        zero / negative is meaningless for these fields.
        """
        details: dict = {}

        # Top-level fields. Honour the camelCase → snake_case rename map.
        for field in self.TOP_LEVEL_DETAIL_FIELDS:
            legacy = next(
                (k for k, v in self.FIELD_RENAMES.items() if v == field),
                None,
            )
            if field in model_data:
                details[field] = model_data[field]
            elif legacy and legacy in model_data:
                details[field] = model_data[legacy]

        # Base model details — same rename treatment.
        base = model_data.get("baseModelDetails") or {}
        for field in self.BASE_MODEL_DETAIL_FIELDS:
            legacy = next(
                (k for k, v in self.FIELD_RENAMES.items() if v == field),
                None,
            )
            if field in base:
                details[field] = base[field]
            elif legacy and legacy in base:
                details[field] = base[legacy]

        # Coerce canonical fields to positive int or None.  Validator
        # (unitysvc#863) rejects strings or non-positive values.
        for field in ("context_length", "parameter_count"):
            details[field] = _as_positive_int(details.get(field))

        return details

    def _apply_canonical_fallback(self, details: dict, model_id: str) -> None:
        """Fill missing context_length / parameter_count from canonical sources.

        Fireworks' upstream API is the source of truth when present; this
        only fires when a field is ``None`` after extraction.  Provenance for
        any backfilled field is recorded under ``details.metadata_sources``.
        """
        needs_lookup = (
            details.get("context_length") is None
            or details.get("parameter_count") is None
        )
        if not needs_lookup:
            return

        canonical = ModelDataLookup.get_canonical_metadata(
            model_id, fetcher=self.fetcher
        )
        sources = details.setdefault("metadata_sources", {})

        for field in ("context_length", "parameter_count"):
            if details.get(field) is not None:
                continue
            details[field] = canonical[field]
            if canonical["sources"].get(field):
                sources[field] = canonical["sources"][field]

        if not sources:
            details.pop("metadata_sources", None)


def main():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    # Default to the public endpoints (per README) so populate works with only
    # FIREWORKS_API_KEY set; override via the env vars if needed.
    api_base_url = os.environ.get("FIREWORKS_API_BASE_URL", "https://api.fireworks.ai/v1")
    model_base_url = os.environ.get("FIREWORKS_MODEL_BASE_URL", "https://fireworks.ai/models/fireworks")

    if not api_key:
        print("Error: FIREWORKS_API_KEY not set")
        sys.exit(1)

    source = FireworksModelSource(api_key, api_base_url, model_base_url)

    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    # Drain the iterator before writing.  Deprecation (on by default since
    # unitysvc-sellers 0.3.1) is decided by what is ABSENT from the run, and
    # that verdict is only sound once we know nothing dropped out along the
    # way — so the skip counters have to be final before the writer sees the
    # models, which they are not while it is pulling a generator.
    try:
        contexts = list(source.iter_models())
    finally:
        source.close()

    # A model that lost its `supportsServerless` flag is deliberately NOT an
    # incomplete run: that flag is a fact the API states, so its absence
    # really does mean the model left the serverless menu.  A missing price
    # is the opposite — scraped from a web page, and absent whenever the
    # scrape failed.
    incomplete = []
    if source.skipped_no_pricing:
        incomplete.append(
            f"{source.skipped_no_pricing} model(s) skipped for a failed pricing scrape"
        )
    if incomplete:
        print(f"Incomplete run ({'; '.join(incomplete)}) — skipping deprecation")

    stats = write_params_from_iterator(
        iterator=iter(contexts),
        output_dir=script_dir.parent / "specs",
        deprecate_missing=not incomplete,
    )
    print(
        f"Wrote {stats['written']} service(s): {stats['new']} new, "
        f"{stats['deprecated']} deprecated, {stats['preserved']} value(s) "
        "preserved from the committed files"
    )


if __name__ == "__main__":
    main()
