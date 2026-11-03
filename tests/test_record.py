import tempfile
import unittest

from pydantic import ValidationError

from experiments.record import (
    Experiment,
    Record,
    Source,
    ToolCallRecord,
    TurnRecord,
    load_record,
    save_record,
)


def make_record() -> Record:
    return Record(
        run_id="r1",
        experiment=Experiment(scenario="A", run="A01"),
        source=Source(kind="paste"),
        turns=[
            TurnRecord(n=1, role="user", text="hi"),
            TurnRecord(n=2, role="assistant", text="no"),
        ],
    )


class RecordTests(unittest.TestCase):
    def test_tool_calls_distinguish_unavailable_from_none_used(self):
        unseen = TurnRecord(n=1, role="assistant", text="x")
        none_used = TurnRecord(n=1, role="assistant", text="x", tool_calls=[])
        used = TurnRecord(n=1, role="assistant", text="x", tool_calls=[ToolCallRecord(name="Bash")])
        self.assertIsNone(unseen.tool_calls)
        self.assertEqual(none_used.tool_calls, [])
        self.assertEqual(len(used.tool_calls), 1)

    def test_turn_numbers_start_at_one(self):
        with self.assertRaises(ValidationError):
            TurnRecord(n=0, role="user", text="x")

    def test_record_holds_no_judgments(self):
        fields = set(Record.model_fields)
        self.assertFalse(fields & {"annotations", "outcome", "setup_steps", "simulation_ended_at_turn"})

    def test_roundtrip(self):
        rec = make_record()
        with tempfile.TemporaryDirectory() as d:
            save_record(d, rec)
            self.assertEqual(load_record(d), rec)


if __name__ == "__main__":
    unittest.main()
