"""Wikitext parser for Escape from Tarkov quest pages."""

import logging
import re
import urllib.parse
from typing import Dict, List, Optional, Tuple
import mwparserfromhell

from src.models import Quest

logger = logging.getLogger(__name__)


class QuestParser:
    """Parses raw MediaWiki wikitext into structured Quest objects."""

    @staticmethod
    def _clean_text(node_or_text) -> str:
        """Strip wiki markup, templates, and HTML tags, returning clean text."""
        if not node_or_text:
            return ""
        parsed = mwparserfromhell.parse(str(node_or_text))
        text = parsed.strip_code()
        # Remove leftover HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _generate_quest_id(title: str) -> str:
        """Generate a consistent, clean quest ID slug."""
        clean = re.sub(r"[^\w\s-]", "", title).strip().lower()
        return re.sub(r"[-\s]+", "_", clean)

    def parse(self, title: str, wikitext: str) -> Optional[Quest]:
        """Parse wikitext for a given quest into a Quest instance."""
        if not wikitext:
            return None

        try:
            code = mwparserfromhell.parse(wikitext)
        except Exception as e:
            logger.error(f"Failed to parse wikitext for '{title}': {e}")
            return None

        # 1. Locate and parse Infobox quest
        infobox = self._extract_infobox(code)
        if not infobox and "category:quests" not in wikitext.lower():
            logger.debug(f"Skipping '{title}': No quest infobox found and not in Category:Quests")
            return None

        trader = infobox.get("given by", {}).get("text")
        location = infobox.get("location", {}).get("text")

        # 2. Extract prerequisites and leads_to from infobox
        prerequisites = self._clean_quest_links(
            infobox.get("previous", {}).get("links", []), current_title=title
        )
        leads_to = self._clean_quest_links(
            infobox.get("leads to", {}).get("links", []), current_title=title
        )

        # 3. Kappa status
        is_kappa_required = self._parse_kappa_status(infobox.get("reqkappa", {}).get("text", ""))

        # 4. Trader Loyalty from Infobox (e.g. |LL requirement = 1)
        req_loyalty: Dict[str, int] = {}
        ll_req_str = infobox.get("ll requirement", {}).get("text", "")
        if ll_req_str.isdigit() and trader:
            req_loyalty[trader] = int(ll_req_str)

        # 5. Extract sections
        sections = self._extract_sections(code)

        # 6. Parse requirements section
        req_bullets, req_player_lvl, section_loyalty, extra_prereqs = self._parse_requirements(
            sections.get("Requirements"), default_trader=trader, current_title=title
        )
        # Merge loyalty requirements
        for t_name, lvl in section_loyalty.items():
            req_loyalty[t_name] = lvl

        # Merge extra prerequisites from text if not already present
        for p in extra_prereqs:
            if p not in prerequisites and p.lower() != title.lower():
                prerequisites.append(p)

        # 7. Parse objectives
        objectives = self._parse_objectives(sections.get("Objectives"))

        # 8. Parse rewards and unlocks
        rewards, unlocks = self._parse_rewards_and_unlocks(sections.get("Rewards"))

        # Construct Wiki URL
        url_encoded_title = urllib.parse.quote(title.replace(" ", "_"), safe=":/_()")
        wiki_url = f"https://escapefromtarkov.fandom.com/wiki/{url_encoded_title}"

        return Quest(
            quest_id=self._generate_quest_id(title),
            name=title,
            wiki_url=wiki_url,
            trader=trader if trader else None,
            location=location if location else None,
            required_player_level=req_player_lvl,
            required_loyalty_level=req_loyalty,
            prerequisites=prerequisites,
            leads_to=leads_to,
            is_kappa_required=is_kappa_required,
            requirements=req_bullets,
            objectives=objectives,
            rewards=rewards,
            unlocks=unlocks,
        )

    def _extract_infobox(self, code: mwparserfromhell.wikicode.Wikicode) -> Dict[str, dict]:
        """Extract key-value pairs and links from the {{Infobox quest}} template."""
        data = {}
        for template in code.filter_templates():
            if template.name.strip().lower() == "infobox quest":
                for param in template.params:
                    key = str(param.name).strip().lstrip("|").lower()
                    text = self._clean_text(param.value)
                    links = [str(link.title).strip() for link in param.value.filter_wikilinks()]
                    data[key] = {"text": text, "links": links}
                break
        return data

    def _extract_sections(
        self, code: mwparserfromhell.wikicode.Wikicode
    ) -> Dict[str, mwparserfromhell.wikicode.Wikicode]:
        """Split wikicode into top-level sections by header."""
        sections = {}
        for section in code.get_sections(levels=[2]):
            headings = section.filter_headings()
            if headings:
                header_name = str(headings[0].title).strip()
                sections[header_name] = section
        return sections

    @staticmethod
    def _clean_quest_links(links: List[str], current_title: Optional[str] = None) -> List[str]:
        """Filter out non-quest wiki links (such as anchors or generic pages)."""
        valid = []
        ignored_names = {
            "see requirements",
            "quests",
            "prapor",
            "therapist",
            "skier",
            "peacekeeper",
            "mechanic",
            "ragman",
            "jaeger",
            "fence",
        }
        if current_title:
            ignored_names.add(current_title.lower())

        for link in links:
            clean = link.split("#")[0].strip()
            if clean and clean.lower() not in ignored_names and clean not in valid:
                valid.append(clean)
        return valid

    @staticmethod
    def _parse_kappa_status(text: str) -> Optional[bool]:
        """Parse whether the quest is required for the Collector / Kappa container."""
        lower = text.lower()
        if "yes" in lower:
            return True
        elif "no" in lower:
            return False
        return None

    def _parse_requirements(
        self,
        section: Optional[mwparserfromhell.wikicode.Wikicode],
        default_trader: Optional[str] = None,
        current_title: Optional[str] = None,
    ) -> Tuple[List[str], Optional[int], Dict[str, int], List[str]]:
        """Extract player level, trader loyalty requirements, and prerequisite quest links."""
        bullets: List[str] = []
        player_level: Optional[int] = None
        loyalty_levels: Dict[str, int] = {}
        extra_prereqs: List[str] = []

        if not section:
            return bullets, player_level, loyalty_levels, extra_prereqs

        lines = str(section).split("\n")
        in_complete_quests_list = False

        for raw_line in lines:
            line = raw_line.strip()
            if not line.startswith("*"):
                in_complete_quests_list = False
                continue

            cleaned = self._clean_text(line.lstrip("*"))
            if not cleaned:
                continue

            bullets.append(cleaned)

            # Player level check: "Must be level 15 to start this quest."
            lvl_match = re.search(r"(?:reach\s+level|be\s+level|lvl)\s*(\d+)", cleaned, re.IGNORECASE)
            if lvl_match and "loyalty" not in cleaned.lower():
                try:
                    player_level = int(lvl_match.group(1))
                except ValueError:
                    pass

            # Trader loyalty check: "Loyalty Level 4 with Ragman"
            ll_match = re.search(
                r"loyalty level\s*(\d+)\s*(?:with\s+([A-Za-z]+))?", cleaned, re.IGNORECASE
            )
            if ll_match:
                trader_name = ll_match.group(2) or default_trader or "Trader"
                try:
                    loyalty_levels[trader_name] = int(ll_match.group(1))
                except ValueError:
                    pass

            # Prerequisite quest list inside requirements text (e.g. Collector)
            if "complete the quest" in cleaned.lower():
                in_complete_quests_list = True

            line_wikicode = mwparserfromhell.parse(raw_line)
            for link in line_wikicode.filter_wikilinks():
                title = str(link.title).strip().split("#")[0]
                if title and title not in ("Quests", "Scavs", default_trader):
                    if current_title and title.lower() == current_title.lower():
                        continue
                    if in_complete_quests_list or line.startswith("**"):
                        if title not in extra_prereqs:
                            extra_prereqs.append(title)

        return bullets, player_level, loyalty_levels, extra_prereqs

    def _parse_objectives(
        self, section: Optional[mwparserfromhell.wikicode.Wikicode]
    ) -> List[str]:
        """Extract and clean objective items."""
        objectives: List[str] = []
        if not section:
            return objectives

        for line in str(section).split("\n"):
            line = line.strip()
            if line.startswith("*"):
                cleaned = self._clean_text(line.lstrip("*"))
                if cleaned and cleaned not in objectives:
                    objectives.append(cleaned)
        return objectives

    def _parse_rewards_and_unlocks(
        self, section: Optional[mwparserfromhell.wikicode.Wikicode]
    ) -> Tuple[List[str], List[str]]:
        """Split reward bullet points into numerical/item rewards and unlock benefits."""
        rewards: List[str] = []
        unlocks: List[str] = []

        if not section:
            return rewards, unlocks

        current_faction_tag = ""
        for line in str(section).split("\n"):
            line = line.strip()

            # Track faction subsections (e.g. ====BEAR Exclusive====)
            if line.startswith("===") or line.startswith("===="):
                header = line.strip("=").strip()
                if "bear" in header.lower():
                    current_faction_tag = "[BEAR] "
                elif "usec" in header.lower():
                    current_faction_tag = "[USEC] "
                else:
                    current_faction_tag = ""
                continue

            if line.startswith("*") and not line.startswith("{{"):
                cleaned = self._clean_text(line.lstrip("*"))
                if not cleaned:
                    continue

                full_item = f"{current_faction_tag}{cleaned}" if current_faction_tag else cleaned

                # Detect unlocks vs regular rewards
                if "unlock" in cleaned.lower():
                    if full_item not in unlocks:
                        unlocks.append(full_item)
                else:
                    if full_item not in rewards:
                        rewards.append(full_item)

        return rewards, unlocks
