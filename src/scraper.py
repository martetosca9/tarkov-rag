"""MediaWiki API client for fetching Escape from Tarkov quest pages."""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from curl_cffi import requests

logger = logging.getLogger(__name__)


class WikiScraper:
    """Client for querying the Escape from Tarkov wiki API."""

    API_URL = "https://escapefromtarkov.fandom.com/api.php"

    def __init__(self, raw_cache_dir: str = "data/raw", timeout: int = 15):
        self.raw_cache_dir = Path(raw_cache_dir)
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session(impersonate="chrome")

    @staticmethod
    def _sanitize_filename(title: str) -> str:
        """Create a safe local filename from a wiki page title."""
        # Replace non-alphanumeric chars (excluding hyphens/underscores) with underscore
        clean_name = re.sub(r"[^\w\-.]", "_", title)
        # Append short hash of original title to prevent collisions
        title_hash = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
        return f"{clean_name}_{title_hash}.json"

    def list_quest_titles(self, category: str = "Category:Quests") -> List[str]:
        """Fetch all article titles belonging to Category:Quests (namespace 0)."""
        logger.info(f"Fetching member titles from category: {category}")
        titles = []
        cmcontinue = None

        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": 0,
            "cmlimit": 500,
            "format": "json",
        }

        while True:
            current_params = params.copy()
            if cmcontinue:
                current_params["cmcontinue"] = cmcontinue

            response = self.session.get(
                self.API_URL, params=current_params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            members = data.get("query", {}).get("categorymembers", [])
            for m in members:
                title = m.get("title")
                if title:
                    titles.append(title)

            if "continue" in data and "cmcontinue" in data["continue"]:
                cmcontinue = data["continue"]["cmcontinue"]
            else:
                break

        logger.info(f"Retrieved {len(titles)} quest page titles.")
        return titles

    def fetch_quest_wikitext(
        self, title: str, use_cache: bool = True, max_retries: int = 3
    ) -> Optional[str]:
        """
        Fetch raw wikitext for a given quest title, using local disk cache if available.
        """
        cache_path = self.raw_cache_dir / self._sanitize_filename(title)

        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    return cached.get("wikitext")
            except Exception as e:
                logger.warning(f"Failed to read cache for '{title}': {e}. Re-fetching.")

        params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "format": "json",
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(
                    self.API_URL, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    logger.warning(
                        f"MediaWiki API error for '{title}': {data['error'].get('info')}"
                    )
                    return None

                wikitext = data.get("parse", {}).get("wikitext", {}).get("*")
                if wikitext is None:
                    logger.warning(f"No wikitext found in response for '{title}'")
                    return None

                # Persist to disk cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"title": title, "wikitext": wikitext}, f, ensure_ascii=False, indent=2)

                return wikitext

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed for '{title}': {e}"
                )
                if attempt < max_retries:
                    time.sleep(1.0 * attempt)
                else:
                    logger.error(f"Exhausted retries fetching '{title}'")
                    return None
