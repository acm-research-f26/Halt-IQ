"""Locked 80-test replay. p_stop is never relabeled p_correct."""
import argparse,csv,json,math,statistics,hashlib
from pathlib import Path
import numpy as np
import torch
from .model import StopPredictor,FEATURES,distance
ROOT=Path(__file__).resolve().parent

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def record(item,index,policy,seed=None):
    rr=item['rounds'];prefix=rr[:index+1]
    return {'id':item['id'],'policy':policy,'seed':seed,'rounds':index+1,'answer':rr[index]['answer'],
            'f1':rr[index]['quality'],'em':rr[index]['answer_em'],
            'qwen_tokens':sum(r['tokens'] for r in prefix),'fixed6_qwen_tokens':sum(r['tokens'] for r in rr),
            'writer_tokens':sum(r['writer_tokens'] for r in prefix),'qwen_critic_tokens':sum(r['critic_tokens'] for r in prefix),
            'qwen_calls':2*(index+1),
            'strands_decisions':index+1 if policy in ('replace','add','strands_gate_0.8') else 0,
            'strands_unique_draft_scores':len({r['answer'] for r in prefix}) if policy in ('replace','add','strands_gate_0.8') else 0}

def summarize(rows):
    return {'n':len(rows),'f1':statistics.mean(r['f1'] for r in rows),'em':statistics.mean(r['em'] for r in rows),
            'avg_rounds':statistics.mean(r['rounds'] for r in rows),
            'avg_qwen_tokens':statistics.mean(r['qwen_tokens'] for r in rows),
            'tokens_percent_of_fixed6':100*sum(r['qwen_tokens'] for r in rows)/sum(r['fixed6_qwen_tokens'] for r in rows),
            'avg_strands_decisions':statistics.mean(r['strands_decisions'] for r in rows),
            'avg_unique_strands_scores':statistics.mean(r['strands_unique_draft_scores'] for r in rows)}

def auc(scores,labels):
    s=np.array(scores);y=np.array(labels);p=s[y==1];n=s[y==0]
    return None if not len(p) or not len(n) else float(((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/(len(p)*len(n)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=ROOT/'results');a=parser.parse_args();out=a.results
    all_items=json.loads((ROOT/'inputs/trajectories.json').read_text());test=[x for x in all_items if x['split']=='test'];assert len(test)==80
    train_ids={x['id'] for x in all_items if x['split']=='train'};dev_ids={x['id'] for x in all_items if x['split']=='dev'}
    parity=[];policyrows=[];roundrows=[];summary=[]
    # Require all preselected checkpoints before accessing test metrics.
    for variant in FEATURES:
        for seed in range(5):
            state=torch.load(out/variant/f'seed{seed}/stop_head.pt',weights_only=True)
            assert set(state['train_ids'])==train_ids and set(state['dev_ids'])==dev_ids
            assert state['data_sha256']==hashlib.sha256((ROOT/'inputs/trajectories.json').read_bytes()).hexdigest()
    for variant in FEATURES:
        for seed in range(5):
            checkpoint=out/variant/f'seed{seed}/stop_head.pt';p=StopPredictor(checkpoint);rows=[]
            state=torch.load(checkpoint,weights_only=True)
            if variant=='original':
                old=torch.load(ROOT/f'reference/original_seed{seed}.pt',weights_only=True)
                match=all(torch.equal(state['weights'][k],v) for k,v in old['weights'].items())
                parity.append({'seed':seed,'exact_weight_match':match});assert match,'Original retraining failed parity'
            for item in test:
                probs=[p.predict(item['question'],item['rounds'][:i+1]) for i in range(6)]
                index=next((i for i,v in enumerate(probs) if v>=.5),5)
                rows.append(record(item,index,variant,seed))
                for i,v in enumerate(probs):roundrows.append({'id':item['id'],'seed':seed,'variant':variant,'round':i+1,'p_stop':v,'strands_score':item['rounds'][i]['strands'],'em_for_offline_diagnostics_only':item['rounds'][i]['answer_em']})
            policyrows+=rows
            summary.append({'variant':variant,'seed':seed,'selected_epoch':state['selected_epoch'],'validation_reward':state['dev_reward'],**summarize(rows)})
    base=[]
    for item in test:
        rr=item['rounds']
        for k in range(1,7):base.append(record(item,k-1,f'fixed{k}'))
        base.append(record(item,next((i for i,r in enumerate(rr) if r['strands']>=.8),5),'strands_gate_0.8'))
        base.append(record(item,next((i for i,r in enumerate(rr) if r['critic_approved']),5),'original_critic_gate'))
        ix=next((i for i in range(2,6) if all(distance(rr[j-1]['answer'],rr[j]['answer'])<.06 for j in [i-1,i])),5)
        base.append(record(item,ix,'distance'))
    baseline_summary=[{'policy':p,**summarize([r for r in base if r['policy']==p])} for p in dict.fromkeys(r['policy'] for r in base)]
    aggregate=[]
    for variant in FEATURES:
        ss=[s for s in summary if s['variant']==variant];row={'variant':variant,'seeds':5,'test_questions_per_seed':80}
        for key in ['f1','em','avg_rounds','avg_qwen_tokens','tokens_percent_of_fixed6']:
            row[key+'_mean']=statistics.mean(s[key] for s in ss);row[key+'_seed_sd']=statistics.stdev(s[key] for s in ss)
        aggregate.append(row)
    # Paired question bootstrap; average over five trained seeds first, never pretend n=400.
    paired=[];rng=np.random.default_rng(42);indices=rng.integers(0,80,(5000,80))
    for variant in FEATURES:
        deltas=[]
        for item in test:
            q=item['id'];v=statistics.mean(r['f1'] for r in policyrows if r['id']==q and r['policy']==variant)
            original=statistics.mean(r['f1'] for r in policyrows if r['id']==q and r['policy']=='original')
            deltas.append(v-original)
        ds=np.array(deltas);paired.append({'variant':variant,'mean_f1_difference_vs_original':float(ds.mean()),'ci95':np.percentile(ds[indices].mean(1),[2.5,97.5]).tolist(),'n_independent_questions':80})
    first=[x['rounds'][0] for x in test]
    diagnostic={'scope':'80 local test first drafts only; threshold 0.8 fixed from supplied notebook',
                'strands_auroc_vs_em':auc([r['strands'] for r in first],[r['answer_em'] for r in first]),
                'note':'Controller predicts p_stop, so no p_correct AUROC is claimed.'}
    write_csv(out/'per_question.csv',policyrows);write_csv(out/'per_round_predictions.csv',roundrows)
    write_csv(out/'summary_by_seed.csv',summary);write_csv(out/'summary.csv',aggregate)
    write_csv(out/'baselines.csv',baseline_summary);write_csv(out/'baseline_per_question.csv',base)
    for name,obj in [('original_parity',parity),('paired_f1',paired),('critic_diagnostic',diagnostic)]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2));print('Original weights reproduce exactly for all five seeds.')
if __name__=='__main__':main()
