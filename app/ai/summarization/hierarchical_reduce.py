from app.ai.schemas.meeting_schema import MeetingNotes
from app.core.logger import get_logger

logger = get_logger("meeting_ai")


class HierarchicalReducer:
    def __init__(
        self,
        reduce_text_chain,
        reduce_chain,
        group_size: int = 6,
    ) -> None:
        self.reduce_text_chain = reduce_text_chain
        self.reduce_chain = reduce_chain
        self.group_size = max(2, group_size)

    def reduce(self, partial_summaries: list[str]) -> MeetingNotes:
        summaries = [summary.strip() for summary in partial_summaries if summary and summary.strip()]
        if not summaries:
            return MeetingNotes()

        current = summaries
        level = 0

        while True:
            num_summaries = len(current)
            group_size = max(
                2,
                min(5, num_summaries // 2),
            )
            if num_summaries <= group_size:
                break

            level += 1
            next_level: list[str] = []
            for group in self._chunk(current, group_size):
                combined = "\n\n---\n\n".join(group)
                logger.info(
                    "Hierarchical reduce level %s with %d summaries",
                    level,
                    len(group),
                )
                reduced = self.reduce_text_chain.invoke({"partial_summaries": combined})
                if reduced and reduced.strip():
                    next_level.append(reduced.strip())
            if not next_level:
                break
            current = next_level

        final_combined = "\n\n---\n\n".join(current)
        return self.reduce_chain.invoke({"partial_summaries": final_combined})

    def _chunk(self, items: list[str], size: int) -> list[list[str]]:
        return [items[i : i + size] for i in range(0, len(items), size)]
