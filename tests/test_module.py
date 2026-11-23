import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speaker_labeler import SpeakerLabeler
from workers.asr_client_worker import WlkMessageProcessor, _parse_server_message


class ModuleTests(unittest.TestCase):
    def test_line_metadata(self):
        msg = json.dumps({"lines": [{"text": "안녕하세요", "speaker": 1, "start": 1.25, "end": 2.5}]})
        event = _parse_server_message(msg)[0]
        self.assertEqual(event["speaker_id"], "spk_1")
        self.assertEqual(event["start_ms"], 1250)
        self.assertEqual(event["end_ms"], 2500)
        self.assertTrue(event["is_final"])

    def test_silence_ignored(self):
        msg = json.dumps({"lines": [{"text": "", "speaker": -2, "start": 0, "end": 1}]})
        self.assertEqual(_parse_server_message(msg), [])

    def test_full_snapshot_deduplicated(self):
        msg = json.dumps({"lines": [{"text": "반갑습니다", "speaker": 1, "start": 0.0, "end": 1.0}]})
        processor = WlkMessageProcessor()
        self.assertEqual(len(processor.process(msg)), 1)
        self.assertEqual(len(processor.process(msg)), 0)
        processor.reset()
        self.assertEqual(len(processor.process(msg)), 1)

    def test_identical_partial_snapshot_is_deduplicated(self):
        msg = json.dumps({"buffer_transcription": "인식 중"})
        processor = WlkMessageProcessor()
        self.assertEqual(processor.process(msg)[0]['partial'], '인식 중')
        self.assertEqual(processor.process(msg), [])

    def test_label_mapping(self):
        labeler = SpeakerLabeler()
        labeler.set_label("spk_1", "교수")
        event = labeler.apply({"speaker_id": "spk_1", "text": "설명합니다"})
        self.assertEqual(event["speaker_label"], "교수")
        self.assertEqual(labeler.get_label("spk_2"), "화자 2")
        self.assertEqual(labeler.get_label(None), "화자 미확인")


if __name__ == "__main__":
    unittest.main()
