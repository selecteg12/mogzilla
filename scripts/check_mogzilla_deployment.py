"""Verify hosted mogzilla with the same public example used in its gallery."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def verify(base_url):
    parsed = urlsplit(base_url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('Use an application origin without a query or fragment.')
    base_url = base_url.rstrip('/')

    def read(path, data=None, mime=None):
        headers = {'User-Agent': 'mogzilla-release-check'}
        if mime:
            headers['Content-Type'] = mime
        with urlopen(Request(base_url+path, data=data, headers=headers), timeout=120) as response:
            return response.read()

    html = read('/').decode('utf-8')
    if 'mogzilla' not in html.lower():
        raise RuntimeError('The public page is not mogzilla.')
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html)
    if not assets or any(not read(asset) for asset in assets):
        raise RuntimeError('Frontend assets are missing.')
    model = json.loads(read('/api/holistic/model'))
    if not model.get('ready') or model.get('revision') != '33c59a04cba4' or model.get('active_profile') != 'personal':
        raise RuntimeError('The approved scoring model is unavailable: '+str(model.get('reason')))
    method = json.loads(read('/api/holistic/method'))
    if method['model']['revision'] != model['revision']:
        raise RuntimeError('Method and inference model revisions differ.')
    # This photograph is already explicitly published in the examples gallery.
    photo = ROOT/'frontend/src/assets/examples/example-03.png'
    analysis = json.loads(read('/api/holistic/analyze',photo.read_bytes(),'image/png')).get('analysis')
    if not analysis or len(analysis.get('landmarks',[])) != 478:
        raise RuntimeError('Real face analysis did not return 478 landmarks.')
    rating = analysis['holistic']
    if rating['revision'] != model['revision'] or abs(rating['value']-10) > .05:
        raise RuntimeError('The gallery example does not reproduce the approved model result.')
    return {'url':base_url,'revision':model['revision'],'model_id':model['model_id'],
            'assets':len(assets),'landmarks':len(analysis['landmarks']),
            'regions':len(rating['regions']),'example_score':rating['value']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    args = parser.parse_args()
    print(json.dumps(verify(args.url),ensure_ascii=False,indent=2))
