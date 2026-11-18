"""Streamlit app integration checks; no model downloads required."""
from pathlib import Path
import json
import argparse
import asyncio
from contextlib import nullcontext
from unittest.mock import Mock, patch
from streamlit.testing.v1 import AppTest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--offline-test-loop', action='store_true', help='Mock the unused asyncio loop for synchronous page tests in environments that prohibit local sockets.')
    args = p.parse_args()
    def loop():
        result = Mock(spec=asyncio.AbstractEventLoop)
        result.is_closed.return_value = False
        return result
    context = patch('asyncio.new_event_loop', side_effect=loop) if args.offline_test_loop else nullcontext()
    with context:
        return verify(args.offline_test_loop)


def verify(mocked):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception, [e.value for e in app.exception]
    report = {'project': 'Mars-Rover-Terrain-Segmentation', 'event_loop_mocked': mocked, 'browser_render_verified': False}
    assert len(app.metric) == 4
    report['standalone_app_passed'] = True
    Path('verification/ui_smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
