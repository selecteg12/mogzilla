"""Validate and serve the current mogzilla scoring release."""
import argparse
import json
import os
import re

from backend.app.main import DIST, app, holistic_analyzer


def validate_bundle():
    index = DIST / 'index.html'
    if not index.is_file():
        raise RuntimeError('Built frontend is missing.')
    content = index.read_text(encoding='utf-8')
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', content)
    if not assets or any(not (DIST / path.lstrip('/')).is_file() for path in assets):
        raise RuntimeError('Built frontend assets are missing.')
    if 'mogzilla' not in content.lower():
        raise RuntimeError('The built frontend is not mogzilla.')
    status = holistic_analyzer.status()
    if not status['ready']:
        raise RuntimeError('Scoring model is unavailable: ' + str(status['reason']))
    if status['active_profile'] != 'personal' or status['revision'] != '33c59a04cba4':
        raise RuntimeError('Expected the approved mogzilla scoring revision 33c59a04cba4.')
    return {'frontend': True, 'assets': len(assets), 'model_id': status['model_id'],
            'revision': status['revision'], 'landmarks': status['landmarks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        status = validate_bundle()
        port = int(os.environ.get('PORT', '8000'))
        if not 1 <= port <= 65535:
            raise ValueError('PORT must be between 1 and 65535.')
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Release validation failed: {exc}\n')
    print(json.dumps(status), flush=True)
    if not args.check:
        import uvicorn
        uvicorn.run(app, host='0.0.0.0', port=port, workers=1, limit_concurrency=4)


if __name__ == '__main__':
    main()
