"""Generate only synthetic CPU CLI examples; never fetch or select real data."""
import json
from copy import deepcopy
from pathlib import Path

from rl.eval.heldout import select_heldout
from spec.eval_acceptance import build_plan

raw = Path('docs/evidence/raw')
train = [{'id': 'synthetic-train', 'question': 'synthetic training fixture', 'source': 'synthetic'}]
candidates = [{'id': f'synthetic-p{i}', 'question': f'synthetic heldout question {i}',
               'answer': f'synthetic gold {i}', 'source': 'synthetic'} for i in range(3)]
manifest = select_heldout(train, candidates, count=3, seed=15, synthetic_fixture=True)
identity = {'name': 'synthetic-only-unmeasured', 'sha256': 'a'*64}
requests = [{'request_id': f'synthetic-r{i}', 'prompt_id': f'synthetic-p{i}',
             'batch_id': 'synthetic-b3', 'batch_size': 3, 'slot': i} for i in range(3)]
plan = build_plan(manifest, candidates, protocol=identity, target=identity, draft=identity,
                  requests=requests, synthetic_fixture=True)
valid = {'schema_version': 1, 'plan_sha256': plan['sha256'], 'synthetic_fixture': True,
         'rows': [{'request_id': f'synthetic-r{i}', 'status': 'completed',
                   'spec_accepted_tokens': 6 if i == 0 else 0,
                   'spec_verify_ct': 2 if i == 0 else 3} for i in range(3)]}
incomplete = deepcopy(valid)
incomplete['rows'] = [{'request_id': 'synthetic-r0', 'status': 'failed',
                       'reason': 'synthetic failure with unknown counters'},
                      {'request_id': 'synthetic-r1', 'status': 'not_run',
                       'reason': 'synthetic planned slot not executed'}]
rejection = deepcopy(valid)
rejection['plan_sha256'] = '0'*64
for name, value in [('plan', plan), ('valid-input', valid), ('incomplete-input', incomplete),
                    ('rejection-input', rejection)]:
    (raw / f'task15-cpu-synthetic-{name}.json').write_text(json.dumps(value, indent=2)+'\n')
