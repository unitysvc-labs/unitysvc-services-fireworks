#!/usr/bin/env python3
"""
Template-based update_services.py for Fireworks.

This script yields model dictionaries that are rendered using Jinja2 templates.
Much simpler than the DataBuilder approach - just yield dicts with the data.

Usage: python scripts/update_services_template.py
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

# Will be provided by unitysvc_sellers
from unitysvc_sellers.template_populate import populate_from_iterator


class FireworksModelSource:
    """Fetches model data from Fireworks.ai API and yields template dictionaries."""

    # Fields to extract from API response into details
    TOP_LEVEL_DETAIL_FIELDS = [
        "calibrated",
        "cluster",
        "contextLength",
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

    BASE_MODEL_DETAIL_FIELDS = [
        "checkpointFormat",
        "defaultPrecision",
        "modelType",
        "moe",
        "parameterCount",
        "supportsFireattention",
        "worldSize",
    ]

    def __init__(self, api_key: str, api_base_url: str, model_base_url: str):
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.model_base_url = model_base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        })

    def iter_models(self) -> Iterator[dict]:
        """
        Yield model dictionaries for template rendering.

        Each dict contains all variables needed by offering.json.j2 and listing.json.j2.
        """
        models = self._fetch_all_models()
        total = len(models)

        for i, model in enumerate(models, 1):
            model_name = model.get("name", "")
            if not model_name:
                continue

            short_name = model_name.split("/")[-1]
            print(f"[{i}/{total}] {short_name}")

            # Get detailed model info
            model_data = self._get_model_details(model_name) or {}
            if not model_data.get("deployedModelRefs"):
                print("  Skipped: No serverless deployment")
                continue

            # Extract pricing from web page
            pricing = self._extract_pricing(short_name)
            if not pricing:
                print("  Skipped: No pricing found")
                continue

            # Determine service type
            service_type = self._determine_service_type(model_name, pricing)
            is_flux = "flux" in model_name.lower()

            # Build details dict from API response
            details = self._extract_details(model_data)

            # Derive capabilities from service type and model details
            capabilities = self._derive_capabilities(service_type, model_data)

            # Yield the template variables
            yield {
                # Required - used for directory name
                "name": short_name,

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
                "pricing": pricing,
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

    def _derive_capabilities(self, service_type: str, model_data: dict) -> list[str]:
        """Derive capabilities from service type and model details.

        Uses HuggingFace-style capability names for consistency across providers.
        """
        caps: list[str] = []

        if service_type == "llm":
            caps.append("llm")
            caps.append("text-generation")
        elif service_type == "image_generation":
            caps.append("text-to-image")
        elif service_type == "embedding":
            caps.append("embedding")
        elif service_type == "prerecorded_transcription":
            caps.append("automatic-speech-recognition")

        # Add capabilities from model details
        if model_data.get("supportsTools"):
            caps.append("tools")
        if model_data.get("supportsImageInput"):
            caps.append("vision")
            caps.append("image-text-to-text")

        return caps

    def _extract_details(self, model_data: dict) -> dict:
        """Extract details dict from API response."""
        details = {}

        # Top-level fields
        for field in self.TOP_LEVEL_DETAIL_FIELDS:
            if field in model_data:
                details[field] = model_data[field]

        # Base model details
        if "baseModelDetails" in model_data:
            for field in self.BASE_MODEL_DETAIL_FIELDS:
                if field in model_data["baseModelDetails"]:
                    details[field] = model_data["baseModelDetails"][field]

        return details


def main():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    api_base_url = os.environ.get("FIREWORKS_API_BASE_URL")
    model_base_url = os.environ.get("FIREWORKS_MODEL_BASE_URL")

    if not api_key:
        print("Error: FIREWORKS_API_KEY not set")
        sys.exit(1)

    source = FireworksModelSource(api_key, api_base_url, model_base_url)

    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    populate_from_iterator(
        iterator=source.iter_models(),
        templates_dir=script_dir.parent / "templates",
        output_dir=script_dir.parent / "services",
    )


if __name__ == "__main__":
    main()
