import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from web_app import app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_serves_demo_page(self):
        response = self.client.get("/")
        self.assertEqual(200, response.status_code)
        self.assertIn("镜台客服", response.text)

    def test_chat_returns_trace_without_calling_external_services(self):
        fake_result = {
            "conversation_id": "demo-api",
            "route": "knowledge_base",
            "intent": {"intent_level1": "usage", "intent_confidence": 0.95},
            "intents": [{"intent_level1": "usage", "intent_confidence": 0.95}],
            "knowledge_docs": [],
            "citations": [],
            "api_data": None,
            "answer": "洁面后使用。",
            "pii_redacted": [],
            "handoff_required": False,
        }
        with patch("web_app.handle_user_question", new=AsyncMock(return_value=fake_result)):
            response = self.client.post(
                "/api/chat",
                json={
                    "conversation_id": "demo-api",
                    "question": "护肤流程是什么？",
                    "messages": [],
                },
            )
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("knowledge_base", payload["route"])
        self.assertIn("request_id", payload["trace"])


if __name__ == "__main__":
    unittest.main()
