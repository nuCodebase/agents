import pytest

from agent_api import ConversationStore


class FakeAgent:
    pass


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_conversations_are_owned_by_user() -> None:
    store = ConversationStore(lambda supervisor, worker: FakeAgent(), max_conversations=2)
    conversation_id, _ = await store.create("alice", "supervisor", "worker")

    assert (await store.get(conversation_id, "alice")).user_id == "alice"
    with pytest.raises(KeyError):
        await store.get(conversation_id, "bob")


@pytest.mark.anyio
async def test_store_evicts_oldest_conversation() -> None:
    store = ConversationStore(lambda supervisor, worker: FakeAgent(), max_conversations=1)
    old_id, _ = await store.create("alice", "supervisor", "worker")
    new_id, _ = await store.create("bob", "supervisor", "worker")

    with pytest.raises(KeyError):
        await store.get(old_id, "alice")
    assert (await store.get(new_id, "bob")).user_id == "bob"
