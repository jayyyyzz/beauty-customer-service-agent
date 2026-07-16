import unittest

from agent_observability import (
    finish_agent_trace,
    record_stage,
    reset_agent_trace,
    start_agent_trace,
)


class AgentTraceTests(unittest.TestCase):
    def test_collects_stage_latency_and_tokens(self):
        token = start_agent_trace("trace-test", "conv-test")
        try:
            record_stage(
                "intent_recognition",
                12.34,
                prompt_tokens=100,
                completion_tokens=20,
                cache_hit_tokens=64,
                cache_miss_tokens=36,
            )
            record_stage("business_tool", 1.2)
            trace = finish_agent_trace(route="business_api")
        finally:
            reset_agent_trace(token)

        self.assertEqual("trace-test", trace["trace_id"])
        self.assertEqual("business_api", trace["route"])
        self.assertEqual(120, trace["tokens"]["total"])
        self.assertEqual(64, trace["tokens"]["cache_hit"])
        self.assertEqual(2, len(trace["stages"]))


if __name__ == "__main__":
    unittest.main()
