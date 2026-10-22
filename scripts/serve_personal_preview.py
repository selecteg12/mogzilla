"""Serve a candidate personal head locally without replacing active weights."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8006)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535.')
    from backend.app import main as api
    from backend.ml.holistic import HolisticAnalyzer
    import uvicorn

    api.holistic_analyzer = HolisticAnalyzer(args.model_dir.resolve(), personal=True)
    status = api.holistic_analyzer.status()
    if not status['ready']:
        parser.error(status['reason'])
    uvicorn.run(api.app, host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
