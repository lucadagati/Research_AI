"""Offline tests (no Ollama, no network). Run: PYTHONPATH=src python -m unittest discover -s tests"""
import os, sys, tempfile, unittest
from unittest import mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ra import guard, llm, dash, cli


class Guard(unittest.TestCase):
    def test_clean_strips_unknown_ids(self):
        st = {"total": 0, "invalid": 0}
        out = guard.clean({"a": ["see [P1, P9] and [P7]"]}, {"P1", "P2"}, st)
        self.assertEqual(out["a"][0], "see [P1] and ")
        self.assertEqual((st["total"], st["invalid"]), (3, 2))

    def test_valid_ids(self):
        st = {"total": 0, "invalid": 0}
        self.assertEqual(guard.valid_ids(["P1", "P5"], {"P1"}, st), ["P1"])
        self.assertEqual(st["invalid"], 1)

    def test_quote(self):
        paper = {"abstract": "We propose a Transformer based solely on attention mechanisms.", "title": "T"}
        self.assertTrue(guard.quote_ok("a Transformer based solely on attention", paper))
        self.assertFalse(guard.quote_ok("completely invented sentence here", paper))
        self.assertFalse(guard.quote_ok("short", paper))


class LLM(unittest.TestCase):
    def test_extract_json(self):
        self.assertEqual(llm._extract_json('noise {"a": 1} tail'), {"a": 1})
        with self.assertRaises(Exception):
            llm._extract_json("no json")

    def test_choose_unloads_largest_first(self):
        ours = {"a": 10, "b": 30, "c": 5}
        self.assertEqual(llm.choose_unloads(ours, need=20, cap=40), ["b"])
        self.assertEqual(llm.choose_unloads({}, need=1, cap=40), [])

    def test_complete_retries_with_mock(self):
        from pydantic import BaseModel
        class S(BaseModel):
            x: int
        outs = iter(["garbage", '{"x": 3}'])
        with mock.patch.object(llm, "chat", side_effect=lambda *a, **k: next(outs)):
            self.assertEqual(llm.complete("p", S).x, 3)


class Dash(unittest.TestCase):
    def test_rid_rejects_traversal(self):
        for bad in ("..", "../x", "a/b", ""):
            with self.assertRaises(ValueError):
                dash._rid(bad)

    def test_creds_from_env(self):
        with mock.patch.dict(os.environ, {"RA_DASH_USER": "u", "RA_DASH_PASS": "p"}):
            self.assertEqual(dash._creds(), ("u", "p"))


class CLI(unittest.TestCase):
    def test_idea_text_not_treated_as_file(self):
        with mock.patch("ra.run.run_idea") as r, mock.patch.object(sys, "argv", ["ra", "idea", "/etc/nonexistent.txt"]):
            cli.main()
        self.assertEqual(r.call_args[0][0], "/etc/nonexistent.txt")

    def test_idea_file_is_read(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("my idea")
        try:
            with mock.patch("ra.run.run_idea") as r, mock.patch.object(sys, "argv", ["ra", "idea", f.name]):
                cli.main()
            self.assertEqual(r.call_args[0][0], "my idea")
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
