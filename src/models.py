"""Data models for Tarkov quests."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Quest(BaseModel):
    """Structured representation of an Escape from Tarkov quest."""

    quest_id: str = Field(description="Unique snake_case identifier for the quest")
    name: str = Field(description="Full display title of the quest")
    wiki_url: str = Field(description="Direct URL to the source wiki article")
    trader: Optional[str] = Field(default=None, description="Trader who gives the quest")
    location: Optional[str] = Field(default=None, description="Map location(s) for the quest")
    required_player_level: Optional[int] = Field(
        default=None, description="Minimum player level required to accept quest"
    )
    required_loyalty_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Trader loyalty level requirements (e.g. {'Ragman': 4})",
    )
    prerequisites: List[str] = Field(
        default_factory=list, description="Names of predecessor quests required to unlock this quest"
    )
    leads_to: List[str] = Field(
        default_factory=list, description="Names of downstream quests unlocked by completing this quest"
    )
    is_kappa_required: Optional[bool] = Field(
        default=None, description="Whether completion is required for Kappa Container"
    )
    requirements: List[str] = Field(
        default_factory=list, description="Raw bullet items from the requirements section"
    )
    objectives: List[str] = Field(
        default_factory=list, description="Step-by-step quest objectives"
    )
    rewards: List[str] = Field(
        default_factory=list, description="Rewards granted upon completion (EXP, cash, rep, items)"
    )
    unlocks: List[str] = Field(
        default_factory=list, description="Items, barters, or crafts unlocked by completing the quest"
    )

    def to_searchable_text(self) -> str:
        """Construct a dense, structured markdown document for vector embeddings."""
        lines = [f"# Quest: {self.name}"]
        if self.trader:
            lines.append(f"Trader: {self.trader}")
        if self.location:
            lines.append(f"Location: {self.location}")
        if self.required_player_level:
            lines.append(f"Required Player Level: {self.required_player_level}")
        if self.required_loyalty_level:
            loyalty_str = ", ".join(f"{t} LL{lvl}" for t, lvl in self.required_loyalty_level.items())
            lines.append(f"Required Trader Loyalty: {loyalty_str}")
        if self.prerequisites:
            lines.append(f"Prerequisites (Previous Quests): {', '.join(self.prerequisites)}")
        if self.leads_to:
            lines.append(f"Leads To (Next Quests): {', '.join(self.leads_to)}")
        if self.is_kappa_required is not None:
            lines.append(f"Required for Kappa: {'Yes' if self.is_kappa_required else 'No'}")

        if self.requirements:
            lines.append("\nRequirements:")
            for req in self.requirements:
                lines.append(f"- {req}")

        if self.objectives:
            lines.append("\nObjectives:")
            for obj in self.objectives:
                lines.append(f"- {obj}")

        if self.rewards:
            lines.append("\nRewards:")
            for rew in self.rewards:
                lines.append(f"- {rew}")

        if self.unlocks:
            lines.append("\nUnlocks:")
            for unl in self.unlocks:
                lines.append(f"- {unl}")

        lines.append(f"\nWiki Source: {self.wiki_url}")
        return "\n".join(lines)
