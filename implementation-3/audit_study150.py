"""Post-run integrity checks and analysis; never changes trained policies."""
import json,csv,collections,statistics,sys,importlib.util
from pathlib import Path
import study
root=study.ROOT/'study150';out=root/'results'
if not (out/'all_seeds.json').exists():raise SystemExit('Wait for all five seed evaluations to finish')
questions={x['id']:x for x in json.loads((root/'questions.json').read_text())}
items=[json.loads(p.read_text()) for p in (root/'cache').glob('hotpot_*.json')]
assert len(items)==150 and {x['id'] for x in items}==set(questions)
assert collections.Counter(x['split'] for x in items)=={'train':50,'dev':20,'test':80}
sys.modules['ujson']=json
spec=importlib.util.spec_from_file_location('official',study.ROOT/'study/official_hotpot_evaluate_v1.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
for item in items:
 assert len(item['rounds'])==6 and item['split']==questions[item['id']]['split']
 for r in item['rounds']:
  assert abs(official.f1_score(r['answer'],questions[item['id']]['ground_truth'])[0]-r['quality'])<1e-12
  assert official.exact_match_score(r['answer'],questions[item['id']]['ground_truth'])==r['answer_em']
  assert sum(r[role]['usage']['prompt_tokens']+r[role]['usage']['completion_tokens'] for role in ['writer','critic'])==r['tokens']
test=[x for x in items if x['split']=='test'];summaries=[];allrows=[]
for seed in range(5):
 data=json.loads((out/f'seed{seed}/evaluation_detailed.json').read_text())
 assert len(data['rows'])==80*7
 for row in data['rows']:
  assert abs(row['total_tokens']-row['writer_tokens']-row['critic_tokens'])<1e-9
  assert row['total_tokens']==row['prompt_tokens']+row['completion_tokens']
  row['ground_truth']=questions[row['id']]['ground_truth'];row['question_type']=questions[row['id']]['type']
  allrows.append(row)
 summaries.extend(dict(seed=seed,**s) for s in data['summary'])
for filename,rows in [('all_seeds_per_question.csv',allrows),('all_seeds_summary.csv',summaries)]:
 with (out/filename).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
logs=[json.loads(line) for line in (root/'cache/calls.jsonl').read_text().splitlines()];ok=[x for x in logs if x['status']=='ok']
audit={'questions':len(items),'rounds':sum(len(x['rounds']) for x in items),'successful_logged_calls':len(ok),'failed_logged_calls':len(logs)-len(ok),
 'all_successful_collection_tokens':sum(x['response']['usage']['total_tokens'] for x in ok),'all_successful_request_seconds':sum(x['response']['seconds'] for x in ok),
 'finish_reasons':dict(collections.Counter(x['response']['finish_reason'] for x in ok)),
 'test_type_counts':dict(collections.Counter(x['type'] for x in test)),
 'test_first_round_approved':sum(x['rounds'][0]['critic_approved'] for x in test),
 'test_any_later_improvement':sum(max(r['quality'] for r in x['rounds'])>x['rounds'][0]['quality'] for x in test),
 'test_final_vs_first':dict(collections.Counter('better' if x['rounds'][-1]['quality']>x['rounds'][0]['quality'] else 'worse' if x['rounds'][-1]['quality']<x['rounds'][0]['quality'] else 'same' for x in test)),
 'learned_stop_histograms':{str(seed):dict(collections.Counter(str(r['rounds']) for r in allrows if r['seed']==seed and r['policy']=='learned')) for seed in range(5)},
 'score_parity':'All 900 rounds match official HotpotQA answer scorer',
 'note':'Collection totals include full six-round generation for all splits, not policy savings; replay timing remains a prefix proxy.'}
study.save(out/'integrity_and_diagnostics.json',audit)
print(json.dumps(audit,indent=2))
