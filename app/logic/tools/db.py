from app.domain.topics import Topic
from app.infra.db.repos.topics import TopicsRepo


class DBTools:
    def __init__(self, topics_repo: TopicsRepo):
        self.topics_repo = topics_repo

    async def get_available_topics(self) -> list[Topic]:
        """
        Retrieve all active topics available in the knowledge base.

        Use this tool before calling search_docs when the user's query relates to a
        specific subject area. The returned topics can then be passed to search_docs
        as the `topics` filter to narrow results.

        Returns:
            list[Topic]: A list of Topic objects, each with an `id` and `name` field.
        """
        return await self.topics_repo.search(archived=False, is_active=True)