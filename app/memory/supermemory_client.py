import os
from supermemory import supermemory

class SupermemoryClient:
    def __init__(self):
        self.client = supermemory(
            api_key=os.environ.get("SUPERMEMORY_API_KEY"),
            base_url=os.environ.get("MEMORY_SERVICE_URL", None)
        )

    def add_memory(self, content, **kwargs):
        return self.client.memories.add(content=content, **kwargs)

    def search_memories(self, query, **kwargs):
        return self.client.search.execute(q=query, **kwargs) 