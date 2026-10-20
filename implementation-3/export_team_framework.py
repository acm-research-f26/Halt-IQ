"""Validate local results with the shared schema; never pool different cohorts."""
import sys,json,csv
from pathlib import Path
R=Path(__file__).resolve().parent;T=R.parent/'.team-reference/vedanshi';Y=R.parent/'.team-reference/Yash'
sys.path.insert(0,str(T/'eval-framework'));from schema import write_rows,read_rows
from compare import policy_table
sys.path.insert(0,str(Y));from haltiq.metrics import hotpot_evaluate
O=R/'study150/results/reference_alignment';O.mkdir(exist_ok=True)
rows=[]
for r in csv.DictReader((R/'study150/results/seed0/per_question.csv').open()):
 c=json.loads((R/'study150/cache'/f"{r['id']}.json").read_text());score=hotpot_evaluate(r['answer'],[c['ground_truth']])
 assert abs(score['token_f1']-float(r['answer_f1']))<1e-10 and score['exact_match']==bool(float(r['answer_em']))
 rows.append(dict(question_id=r['id'].replace('hotpot_','hotpotqa:'),split='test',policy=r['policy'],source='sri-local80-seed0-replay',final_answer=r['answer'],rounds=int(r['rounds']),total_tokens=int(r['total_tokens']),seconds=float(r['request_seconds']),exact_match=score['exact_match'],token_f1=score['token_f1'],info_score=None,cost_usd=0.0))
 if r['policy']=='fixed_1':
  w=c['rounds'][0]['writer'];rows.append(dict(rows[-1],policy='one-draft-writer-only',total_tokens=w['usage']['total_tokens'],seconds=w['seconds']))
write_rows(O/'local80_team_schema.jsonl',rows);lines,stats=policy_table(read_rows(O/'local80_team_schema.jsonl'),'fixed_6')
(O/'framework_comparison.md').write_text('# Local 80 only — shared framework output\n\nReplay timings, not measured live savings. Six-round local loop differs from team three-round loop.\n\n'+'\n'.join(lines))
local=json.loads((R/'study150/questions.json').read_text());team=[json.loads(l) for l in (Y/'data/hotpotqa/tasks.jsonl').read_text().splitlines() if l.strip()]
bench=list(csv.DictReader((T/'strands-reranker/results/haltiq_strands_experiments2.csv').open()))
canon=lambda s:' '.join(s.lower().split())
audit={}
for partition in ['train','dev','test']:
 q=[r for r in local if r['split']==partition];ids={r['id'].split('_')[-1] for r in q};texts={canon(r['question']) for r in q}
 hold=[r for r in team if r['split']=='test'];audit[partition]={'n':len(q),'team_holdout_id_overlap':sorted(ids&{r['id'].split(':')[-1] for r in hold}),'team_holdout_text_overlap':sorted(texts&{canon(r['question']) for r in hold}),'benchmark_id_overlap':sorted(ids&{r['id'] for r in bench})}
(O/'overlap_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2));print('Validated',len(rows),'rows')
