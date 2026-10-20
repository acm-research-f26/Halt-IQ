"""Auditable expanded HotpotQA study; original pilot code and results stay intact."""
import argparse, csv, hashlib, json, math, random, time, urllib.request
from pathlib import Path
from types import SimpleNamespace
import torch
import hotpot as h
ROOT=Path(__file__).resolve().parent
STUDY=ROOT/'study'
MODEL='mlx-community/Qwen2.5-1.5B-Instruct-4bit'
WRITER='Answer the question using the supplied passages. Return ONLY the short final answer, without explanation. Revise using the critique if present.'
CRITIC='Check whether the answer is supported by the supplied passages and fully answers the question. Return APPROVED if so. Otherwise give one brief correction.'
POLICIES=['fixed_1','fixed_3','fixed_6','critic','distance','learned','oracle_f1']
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def prepare(source):
 import pyarrow.parquet as pq
 old=json.loads((ROOT/'data/scenarios_hotpot.json').read_text())
 excluded={x['id'].removeprefix('hotpot_') for x in old}
 rows=pq.read_table(source).to_pylist()
 eligible=[r for r in rows if r['id'] not in excluded and len(r['context']['title'])==10]
 rng=random.Random(20261016);rng.shuffle(eligible)
 selected=eligible[:600];assert len(selected)==600
 result=[]
 for i,r in enumerate(selected):
  c=r['context'];contexts=[title+': '+''.join(sentences) for title,sentences in zip(c['title'],c['sentences'])]
  assert len(contexts)==10
  result.append({'id':'hotpot_'+r['id'],'split':'train' if i<300 else 'dev' if i<400 else 'test',
   'question':r['question'],'contexts':contexts,'ground_truth':r['answer'],'type':r['type'],'level':r['level']})
 target=STUDY/'questions.json'
 if target.exists() and json.loads(target.read_text())!=result:raise ValueError('Refusing to replace a different dataset')
 save(target,result)
 save(STUDY/'protocol.json',{'version':1,'date':'2026-10-16','seed':20261016,'questions':600,'split_counts':{'train':300,'dev':100,'test':200},
  'source':'https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet',
  'source_sha256':digest(source),'dataset_sha256':digest(target),'excluded_pilot_ids':sorted(excluded),
  'eligibility':'Exactly 10 supplied passages; prior 80 pilot IDs excluded',
  'partition_note':'All partitions are local partitions of official development data, not official hidden test data. Other team exclusions were unavailable.',
  'model':MODEL,'max_rounds':6,'temperature':0,'max_tokens_per_call':256,'training_seeds':[0,1,2,3,4],
  'cost':0.1,'epochs':20,'group_size':8,'stop_threshold':0.5,'distance_threshold':0.06,
  'primary_comparisons':['learned versus fixed_1','learned versus fixed_6'],'primary_metric':'answer_f1',
  'secondary_metrics':['answer_em','rounds','prompt_tokens','completion_tokens','writer_tokens','critic_tokens','llm_calls','request_seconds'],
  'inference':'paired question bootstrap, 5000 resamples, seed42; repeat policy training across five seeds; no non-inferiority claim',
  'writer_prompt':WRITER,'critic_prompt':CRITIC})
 print('Prepared 600 questions: 300 train / 100 validation / 200 test; excluded all 80 pilot IDs')
def request(endpoint,messages,log,role,qid,round_no):
 body={'model':MODEL,'messages':messages,'temperature':0,'max_tokens':256}
 started=time.perf_counter(); req=urllib.request.Request(endpoint,json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer local'})
 try:
  with urllib.request.urlopen(req,timeout=600) as response: obj=json.load(response)
  elapsed=time.perf_counter()-started;usage=obj.get('usage',{})
  for k in ['prompt_tokens','completion_tokens','total_tokens']:
   if not isinstance(usage.get(k),int) or usage[k]<0:raise ValueError('Missing valid provider token usage: '+k)
  if usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']:raise ValueError('Inconsistent token accounting')
  result={'text':obj['choices'][0]['message']['content'],'usage':usage,'seconds':elapsed,'finish_reason':obj['choices'][0].get('finish_reason')}
  record={'id':qid,'round':round_no,'role':role,'request':body,'response':result,'status':'ok'}
 except Exception as e:
  record={'id':qid,'round':round_no,'role':role,'status':'failed','error':str(e),'seconds':time.perf_counter()-started}
  with log.open('a') as f:f.write(json.dumps(record)+'\n')
  raise
 with log.open('a') as f:f.write(json.dumps(record)+'\n')
 return result

def generate(endpoint,limit):
 questions=json.loads((STUDY/'questions.json').read_text());protocol=json.loads((STUDY/'protocol.json').read_text())
 if digest(STUDY/'questions.json')!=protocol['dataset_sha256']:raise ValueError('Dataset changed')
 out=STUDY/'cache';out.mkdir(exist_ok=True)
 meta={'source':'live','model':MODEL,'endpoint':endpoint,'max_rounds':6,'distance_kind':'lexical_jaccard_proxy','reward_metric':'answer_f1','dataset_sha256':protocol['dataset_sha256'],'protocol_sha256':digest(STUDY/'protocol.json')}
 mp=out/'manifest.json'
 if mp.exists() and json.loads(mp.read_text())!=meta:raise ValueError('Cache mismatch')
 save(mp,meta)
 # Check availability before doing work. This does not start an inference request.
 with urllib.request.urlopen(endpoint.rsplit('/chat/completions',1)[0]+'/models',timeout=5) as response:json.load(response)
 done=0
 for q in questions:
  target=out/(q['id']+'.json');partial=out/(q['id']+'.partial')
  if target.exists():continue
  state=json.loads(partial.read_text()) if partial.exists() else {'rounds':[],'pending_writer':None}
  rounds=state['rounds'];previous=rounds[-1]['answer'] if rounds else '';feedback=rounds[-1]['critique'] if rounds else ''
  for i in range(len(rounds),6):
   writer=state['pending_writer']
   if writer is None:
    writer=request(endpoint,[{'role':'system','content':WRITER},{'role':'user','content':json.dumps({'question':q['question'],'contexts':q['contexts'],'previous_answer':previous,'critique':feedback})}],out/'calls.jsonl','writer',q['id'],i+1)
    state['pending_writer']=writer;save(partial,state)
   answer=writer['text']
   critic=request(endpoint,[{'role':'system','content':CRITIC},{'role':'user','content':json.dumps({'question':q['question'],'contexts':q['contexts'],'answer':answer})}],out/'calls.jsonl','critic',q['id'],i+1)
   feedback=critic['text'];em,f1=h.quality(answer,q['ground_truth'])
   rounds.append({'answer':answer,'critique':feedback,'distance':None if i==0 else h.lexical_distance(previous,answer),'critic_approved':feedback.strip()=='APPROVED',
    'quality':f1,'answer_em':em,'tokens':writer['usage']['total_tokens']+critic['usage']['total_tokens'],'writer':writer,'critic':critic})
   state['pending_writer']=None;save(partial,state);previous=answer
  save(target,{'id':q['id'],'question':q['question'],'ground_truth':q['ground_truth'],'type':q['type'],'split':q['split'],'source':'live','rounds':rounds})
  partial.unlink();done+=1;print(f"Cached {q['id']} ({q['split']}); {done} completed this invocation",flush=True)
  if limit and done>=limit:break
 print('Generation invocation complete. Resume with the same command.')

def ci(values):
 rng=random.Random(42);n=len(values)
 boot=sorted(sum(rng.choices(values,k=n))/n for _ in range(5000))
 return [boot[124],boot[4874]]
def rows_for(items,model,seed):
 rows=[]
 for item in items:
  rr=item['rounds']
  for policy in POLICIES:
   t=max(range(len(rr)),key=lambda i:(rr[i]['quality'],-i)) if policy=='oracle_f1' else h.choose(policy,model,rr)
   prefix=rr[:t+1];detailed=all('writer' in r and 'critic' in r for r in prefix)
   row={'id':item['id'],'question':item['question'],'seed':seed,'policy':policy,'answer':rr[t]['answer'],'answer_f1':rr[t]['quality'],'answer_em':rr[t]['answer_em'],
    'rounds':t+1,'total_tokens':sum(r['tokens'] for r in prefix),'llm_calls':2*(t+1),'calls_basis':'two successful calls per cached round; excludes failed generation attempts'}
   for key,fn in {'prompt_tokens':lambda r:sum(r[z]['usage']['prompt_tokens'] for z in ['writer','critic']),
                  'completion_tokens':lambda r:sum(r[z]['usage']['completion_tokens'] for z in ['writer','critic']),
                  'writer_tokens':lambda r:r['writer']['usage']['total_tokens'], 'critic_tokens':lambda r:r['critic']['usage']['total_tokens'],
                  'request_seconds':lambda r:r['writer']['seconds']+r['critic']['seconds']}.items():row[key]=sum(fn(r) for r in prefix) if detailed else None
   rows.append(row)
 return rows

def report(cache,checkpoint,output):
 meta,items=h.load_cache(cache);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if state['cache_manifest']!=meta or state['features']!=h.FEATURES:raise ValueError('Checkpoint mismatch')
 test=[x for x in items if x['split']=='test']
 if not test or {x['id'] for x in test}&set(state['train_ids']+state['dev_ids']):raise ValueError('Invalid test split')
 model=h.StopHead();model.load_state_dict(state['weights']);model.eval()
 with torch.no_grad(): rows=rows_for(test,model,state['seed'])
 summaries=[]
 for policy in POLICIES:
  selected=[r for r in rows if r['policy']==policy];s={'policy':policy,'questions':len(selected)}
  for key in ['answer_f1','answer_em','rounds','total_tokens','llm_calls','prompt_tokens','completion_tokens','writer_tokens','critic_tokens','request_seconds']:
   vals=[r[key] for r in selected];s['mean_'+key]=None if any(v is None for v in vals) else sum(vals)/len(vals)
  s['f1_ci95']=ci([r['answer_f1'] for r in selected]);s['em_ci95']=ci([r['answer_em'] for r in selected])
  for reference in ['fixed_1','fixed_6']:
   base={r['id']:r for r in rows if r['policy']==reference}
   diffs=[r['answer_f1']-base[r['id']]['answer_f1'] for r in selected]
   s['f1_difference_vs_'+reference]=sum(diffs)/len(diffs);s['f1_difference_ci95_vs_'+reference]=ci(diffs)
   s['token_savings_percent_vs_'+reference]=100*(1-sum(r['total_tokens'] for r in selected)/sum(r['total_tokens'] for r in base.values()))
  summaries.append(s)
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 save(out/'evaluation_detailed.json',{'source':meta['source'],'evaluation':'cached trajectory replay, not live stopped execution','seed':state['seed'],'rows':rows,'summary':summaries,
  'oracle_note':'Hindsight maximum F1, ties choose earliest; uses gold and is not deployable. Other metrics on this row describe the F1-optimal prefix, not independent upper bounds.',
  'cost_note':'Writer and critic included through stopping round. Timing is sum of recorded request times, not a measured live policy runtime. Older pilot timing and token breakdown unavailable.'})
 with (out/'per_question.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Audited stopping-policy comparison','',f"Source: {meta['source']}; {len(test)} test questions; seed {state['seed']}. Cached replay.",'','|Policy|F1|EM|Rounds|Tokens|Calls|Saved vs Fixed 6|','|---|---:|---:|---:|---:|---:|---:|']
 for s in summaries:lines.append(f"|{s['policy']}|{s['mean_answer_f1']:.4f}|{s['mean_answer_em']:.4f}|{s['mean_rounds']:.2f}|{s['mean_total_tokens']:.1f}|{s['mean_llm_calls']:.2f}|{s['token_savings_percent_vs_fixed_6']:.2f}%|")
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

def run():
 questions=json.loads((STUDY/'questions.json').read_text());meta,items=h.load_cache(STUDY/'cache')
 if {x['id'] for x in questions}!={x['id'] for x in items}:raise ValueError('All planned trajectories must be complete before final training/evaluation')
 expected={q['id']:q['split'] for q in questions}
 if any(len(x['rounds'])!=6 or x['split']!=expected[x['id']] for x in items):raise ValueError('Trajectory length/split mismatch')
 protocol=json.loads((STUDY/'protocol.json').read_text())
 if meta['protocol_sha256']!=digest(STUDY/'protocol.json') or meta['dataset_sha256']!=digest(STUDY/'questions.json'):raise ValueError('Protocol changed')
 # Finish all seed training before any test scoring; no best-test-seed selection.
 for seed in protocol['training_seeds']:
  out=STUDY/'results'/f'seed{seed}'
  h.train(SimpleNamespace(cache=STUDY/'cache',output=out,seed=seed,epochs=20,group_size=8,cost=.1))
 for seed in protocol['training_seeds']:
  out=STUDY/'results'/f'seed{seed}';report(STUDY/'cache',out/'stop_head.pt',out)
 save(STUDY/'results'/'all_seeds.json',{str(seed):json.loads((STUDY/'results'/f'seed{seed}'/'evaluation_detailed.json').read_text())['summary'] for seed in protocol['training_seeds']})
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('prepare');a.add_argument('--source',required=True)
 a=sub.add_parser('generate');a.add_argument('--endpoint',default='http://127.0.0.1:8080/v1/chat/completions');a.add_argument('--limit',type=int)
 a=sub.add_parser('report');a.add_argument('--cache',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--output',required=True)
 sub.add_parser('run');a=p.parse_args()
 if a.command=='prepare':prepare(a.source)
 elif a.command=='generate':generate(a.endpoint,a.limit)
 elif a.command=='report':report(a.cache,a.checkpoint,a.output)
 else:run()
