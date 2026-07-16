import tempfile
import unittest
from pathlib import Path

from handoff_store import HandoffStore


class HandoffStoreTests(unittest.TestCase):
    def test_creates_persistent_redacted_ticket(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HandoffStore(Path(tmp) / "handoffs.db")
            ticket = store.create(
                conversation_id="demo-1",
                reason="low_confidence",
                summary="请联系我，手机号13812345678",
                context={"email": "u@example.com"},
            )
            loaded = store.get(ticket["ticket_id"])
            self.assertEqual("open", loaded["status"])
            self.assertNotIn("13812345678", loaded["summary"])
            self.assertNotIn("u@example.com", str(loaded["context"]))


if __name__ == "__main__":
    unittest.main()
