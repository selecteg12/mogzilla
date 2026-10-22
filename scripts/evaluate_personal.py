"""Exercise the real API, quality gates, and active personal head on the holdout."""
import argparse
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import main as api
from backend.ml.holistic import HolisticAnalyzer
from scripts.extract_personal_features import training_rgb
from scripts.train_personal import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    root=args.experiment
    data=json.loads((root/'features.json').read_text(encoding='utf-8'))
    predictions=json.loads((root/'predictions.json').read_text(encoding='utf-8'))
    expected={row['id']:row for row in predictions if row['split']=='test'}
    api.holistic_analyzer=HolisticAnalyzer(personal=True)
    client=TestClient(api.app)
    status=client.get('/api/holistic/model').json()
    assert status['ready'] and status['active_profile']=='personal'
    accepted,rejected=[],[]
    for row in data['records']:
        if row['id'] not in expected:continue
        payload=(Path(data['dataset'])/row['path']).read_bytes()
        if row.get('training_normalized'):
            rgb,_=training_rgb(payload)
            stream=io.BytesIO()
            Image.fromarray(rgb).save(stream,format='PNG')
            payload=stream.getvalue()
        response=client.post('/api/holistic/analyze',content=payload,headers={'content-type':'image/png' if row.get('training_normalized') else 'image/jpeg'})
        if response.status_code!=200 or not response.json().get('analysis'):
            rejected.append({'id':row['id'],'status':response.status_code,'detail':response.json()})
            continue
        analysis=response.json()['analysis']
        result=analysis['holistic']
        assert result['kind']=='predicted_personal_rating'
        assert len(analysis['landmarks'])==478
        delta=abs(result['value']-expected[row['id']]['personal'])
        assert delta<.011, (row['id'],delta)
        accepted.append({'id':row['id'],'score':row['score'],'personal':result['value'],'original':result['baseline']['value'],
                         'expected_delta':round(delta,5),'stability':result['stability'],'warnings':analysis['quality']['warnings']})
    report={'active_model':status,'holdout_photos':len(expected),'accepted':len(accepted),'rejected':len(rejected),
            'original':metrics([r['score'] for r in accepted],[r['original'] for r in accepted]),
            'personal':metrics([r['score'] for r in accepted],[r['personal'] for r in accepted]),
            'records':accepted,'rejections':rejected}
    (root/'api-evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in {'records','rejections','active_model'}},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
