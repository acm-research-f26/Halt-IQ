"""Implementation 3 laptop adaptation: frozen HotpotQA trajectories + CPU GRPO stop head."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, os, random, re, string, time
import urllib.request
import torch

ROOT = Path(__file__).resolve().parent
FEATURES = ['distance', 'round_fraction', 'distance_delta', 'critic_approved', 'has_previous_draft']

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')

def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def quality(answer, gold):
    a, b = normalize(answer), normalize(gold)
    em = float(a == b)
    if a in ('yes','no','noanswer') or b in ('yes','no','noanswer'):
        if a != b: return em, 0.0
    aa, bb = a.split(), b.split()
    common = sum((Counter(aa) & Counter(bb)).values())
    if not common: return em, 0.0  # Official HotpotQA F1 is zero for empty token overlap.
    precision, recall = common / len(aa), common / len(bb)
    return em, 2 * precision * recall / (precision + recall)

def split_for(s):
    # Keep the supplied evaluation set held out. Split only the supplied dev pool.
    if s['split'] == 'test': return 'test'
    return 'dev' if int(hashlib.sha256(s['id'].encode()).hexdigest(),16) % 4 == 0 else 'train'

def features(rounds, index):
    r = rounds[index]
    distance = r['distance']
    previous = rounds[index-1]['distance'] if index else None
    return torch.tensor([1.0 if distance is None else distance,
                         (index+1)/len(rounds),
                         0.0 if distance is None or previous is None else distance-previous,
                         float(r['critic_approved']), float(index>0)], dtype=torch.float32)

def lexical_distance(a,b):
    # Explicit CPU proxy, NOT embedding cosine distance or semantic entropy.
    a,b = set(normalize(a).split()),set(normalize(b).split())
    return 1-len(a & b)/max(1,len(a | b))

def chat(endpoint, key, model, messages):
    data = json.dumps({'model':model,'messages':messages,'temperature':0,
                       'max_tokens':256}).encode()
    req = urllib.request.Request(endpoint, data=data,
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=120) as response: result=json.load(response)
    usage=result.get('usage',{})
    if 'total_tokens' not in usage: raise ValueError('Provider must report total_tokens for honest cost accounting')
    return result['choices'][0]['message']['content'], int(usage['total_tokens'])

def generate(args):
    scenarios=json.loads((ROOT/'data/scenarios_hotpot.json').read_text())
    if args.limit: scenarios=scenarios[:args.limit]
    source='synthetic' if args.mock else 'live'
    if not args.mock:
        key=os.environ.get('HALTIQ_API_KEY','')
        if not key or not args.endpoint or not args.model:
            raise ValueError('Live generation requires HALTIQ_API_KEY, --endpoint and --model. No API calls made.')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    metadata={'source':source,'model':args.model,'endpoint':args.endpoint,'max_rounds':args.rounds,
              'distance_kind':'synthetic' if args.mock else 'lexical_jaccard_proxy',
              'reward_metric':'synthetic_quality' if args.mock else 'answer_f1',
              'dataset_sha256':hashlib.sha256((ROOT/'data/scenarios_hotpot.json').read_bytes()).hexdigest()}
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError('Cache configuration differs: choose a new output directory')
    save(manifest,metadata)
    for s in scenarios:
        path=out/(s['id']+'.json')
        if path.exists(): continue
        rounds=[]; previous=''; feedback=''
        rng=random.Random(s['id'])
        peak=rng.randint(1,args.rounds)
        for t in range(args.rounds):
            if args.mock:
                score=max(0,min(1,0.45+0.1*min(t+1,peak)-0.02*max(0,t+1-peak)))
                answer=f'SYNTHETIC draft {t+1}'
                distance=None if t==0 else 0.3*0.45**(t-1)
                approved=t+1>=peak; tokens=280+15*t; em=0.0
            else:
                # Reference answer is intentionally excluded from all model inputs.
                state={'question':s['question'],'contexts':s['contexts'],'previous_answer':previous,'critique':feedback}
                answer,wt=chat(args.endpoint,key,args.model,[
                    {'role':'system','content':'Answer the question using the supplied passages. Return ONLY the short final answer, without explanation. Revise using the critique if present.'},
                    {'role':'user','content':json.dumps(state)}])
                feedback,ct=chat(args.endpoint,key,args.model,[
                    {'role':'system','content':'Check whether the answer is supported by the supplied passages and fully answers the question. Return APPROVED if so. Otherwise give one brief correction.'},
                    {'role':'user','content':json.dumps({'question':s['question'],'contexts':s['contexts'],'answer':answer})}])
                approved=feedback.strip()=='APPROVED'; tokens=wt+ct
                distance=None if t==0 else lexical_distance(previous,answer)
                em,score=quality(answer,s['ground_truth'])
            rounds.append({'answer':answer,'distance':distance,'critic_approved':approved,
                           'tokens':tokens,'quality':score,'answer_em':em})
            previous=answer
        save(path,{'id':s['id'],'split':split_for(s),'question':s['question'],'source':source,'rounds':rounds})
        print(f"Cached {s['id']} ({split_for(s)}, {source})",flush=True)
    print('Cache ready:',out)

def load_cache(path):
    path=Path(path); meta=json.loads((path/'manifest.json').read_text())
    items=[json.loads(p.read_text()) for p in sorted(path.glob('hotpot_*.json'))]
    if not items: raise ValueError('No trajectories found')
    ids=[x['id'] for x in items]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate question IDs')
    for item in items:
        if item['source']!=meta['source']: raise ValueError('Mixed synthetic and real trajectories')
        if item['split'] not in ('train','dev','test') or not item['rounds']: raise ValueError('Invalid trajectory')
        for r in item['rounds']:
            if not math.isfinite(r['quality']) or not 0<=r['quality']<=1 or r['tokens']<0: raise ValueError('Invalid quality/cost')
    return meta,items

class StopHead(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.head=torch.nn.Linear(5,1)
    def forward(self,x): return self.head(x).squeeze(-1)

def rollout(model, rounds, greedy=False):
    decisions=[]
    for t in range(len(rounds)-1):
        x=features(rounds,t); dist=torch.distributions.Bernoulli(logits=model(x))
        action=(dist.probs>=0.5).float() if greedy else dist.sample()
        decisions.append((x,action,dist.log_prob(action).detach()))
        if action.item(): return t,decisions
    return len(rounds)-1,decisions

def reward(rounds,index,cost):
    fraction=sum(r['tokens'] for r in rounds[:index+1])/max(1,sum(r['tokens'] for r in rounds))
    return rounds[index]['quality']-cost*fraction

def train(args):
    meta,items=load_cache(args.cache)
    training=[x for x in items if x['split']=='train']; dev=[x for x in items if x['split']=='dev']
    if not training or not dev: raise ValueError('Need both train and dev questions; generate more scenarios')
    if args.group_size<2: raise ValueError('group-size must be at least 2')
    torch.set_num_threads(1); torch.manual_seed(args.seed); rng=random.Random(args.seed)
    model=StopHead(); opt=torch.optim.Adam(model.parameters(),lr=0.01)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    best=-float('inf'); history=[]
    for epoch in range(args.epochs):
        rng.shuffle(training); signals=0
        for item in training:
            rounds=item['rounds']
            with torch.no_grad(): group=[rollout(model,rounds) for _ in range(args.group_size)]
            returns=torch.tensor([reward(rounds,t,args.cost) for t,_ in group])
            if returns.std(unbiased=False)<1e-8: continue
            advantages=(returns-returns.mean())/(returns.std(unbiased=False)+1e-6)
            losses=[]
            for (_,decisions),adv in zip(group,advantages):
                if not decisions: continue
                current=sum(torch.distributions.Bernoulli(logits=model(x)).log_prob(a) for x,a,_ in decisions)
                old=sum(lp for _,_,lp in decisions)
                ratio=torch.exp(current-old)
                losses.append(-torch.minimum(ratio*adv,ratio.clamp(0.8,1.2)*adv))
            if losses:
                opt.zero_grad(); torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); signals+=1
        with torch.no_grad():
            score=sum(reward(x['rounds'],rollout(model,x['rounds'],True)[0],args.cost) for x in dev)/len(dev)
        history.append({'epoch':epoch+1,'dev_reward':score,'groups_with_signal':signals})
        if score>best:
            best=score
            torch.save({'weights':model.state_dict(),'features':FEATURES,'source':meta['source'],
                        'cache_manifest':meta,'seed':args.seed,'cost':args.cost,
                        'train_ids':[x['id'] for x in training],'dev_ids':[x['id'] for x in dev]},output/'stop_head.pt')
    save(output/'training.json',{'source':meta['source'],'history':history,'seed':args.seed,
                              'architecture':'CPU linear stop head, 6 parameters; no LLM fine-tuning'})
    print(f'Trained CPU stopping head. Best dev reward: {best:.4f}. Source: {meta["source"]}')

def choose(policy,model,rounds):
    if policy.startswith('fixed_'): return min(int(policy.split('_')[1]),len(rounds))-1
    if policy=='learned': return rollout(model,rounds,True)[0]
    if policy=='critic': return next((i for i,r in enumerate(rounds) if r['critic_approved']),len(rounds)-1)
    for i in range(2,len(rounds)):
        if all(r['distance'] is not None and r['distance']<0.06 for r in rounds[i-1:i+1]):return i
    return len(rounds)-1

def evaluate(args):
    meta,items=load_cache(args.cache)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state['features']!=FEATURES or state['cache_manifest']!=meta: raise ValueError('Checkpoint/cache configuration mismatch')
    items=[x for x in items if x['split']=='test']
    if not items: raise ValueError('No held-out test questions')
    if set(x['id'] for x in items)&set(state['train_ids']+state['dev_ids']):raise ValueError('Train/test leakage')
    model=StopHead();model.load_state_dict(state['weights']);model.eval()
    policies=['fixed_1','fixed_3','fixed_6','critic','distance','learned']; rows=[]
    with torch.no_grad():
        for item in items:
            for policy in policies:
                t=choose(policy,model,item['rounds']); rr=item['rounds']
                rows.append({'id':item['id'],'policy':policy,'rounds':t+1,'quality':rr[t]['quality'],
                             'tokens':sum(r['tokens'] for r in rr[:t+1])})
    base={r['id']:r for r in rows if r['policy']=='fixed_6'}; summary=[]
    for policy in policies:
        pp=[r for r in rows if r['policy']==policy];n=len(pp)
        differences=[r['quality']-base[r['id']]['quality'] for r in pp]
        rng=random.Random(42)
        boot=sorted(sum(rng.choices(differences,k=n))/n for _ in range(1000))
        s={'policy':policy,'questions':n,'mean_rounds':sum(r['rounds'] for r in pp)/n,
           'mean_quality':sum(r['quality'] for r in pp)/n,'mean_tokens':sum(r['tokens'] for r in pp)/n,
           'paired_quality_difference_ci95':[boot[25],boot[974]]}
        s['tokens_saved_percent']=100*(1-sum(r['tokens'] for r in pp)/max(1,sum(base[r['id']]['tokens'] for r in pp)))
        summary.append(s)
    out=Path(args.output);save(out/'evaluation.json',{'source':meta['source'],'metric':meta['reward_metric'],'rows':rows,'summary':summary})
    lines=['# HotpotQA stopping comparison',f'\nSource: **{meta["source"]}**. Metric: {meta["reward_metric"]}.',
           '\nSynthetic results are pipeline checks only. No non-inferiority claim is made.',
           '\n| Policy | Rounds | Quality | Tokens saved |','|---|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['policy']} | {s['mean_rounds']:.2f} | {s['mean_quality']:.3f} | {s['tokens_saved_percent']:.1f}% |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--mock',action='store_true');g.add_argument('--endpoint');g.add_argument('--model');g.add_argument('--limit',type=int);g.add_argument('--rounds',type=int,default=6);g.add_argument('--output',required=True)
    t=sub.add_parser('train');t.add_argument('--cache',required=True);t.add_argument('--output',required=True);t.add_argument('--epochs',type=int,default=20);t.add_argument('--group-size',type=int,default=8);t.add_argument('--seed',type=int,default=0);t.add_argument('--cost',type=float,default=0.1)
    e=sub.add_parser('evaluate');e.add_argument('--cache',required=True);e.add_argument('--checkpoint',required=True);e.add_argument('--output',required=True)
    args=p.parse_args()
    if getattr(args,'rounds',6)<2: p.error('rounds must be at least 2')
    if getattr(args,'epochs',1)<1: p.error('epochs must be positive')
    {'generate':generate,'train':train,'evaluate':evaluate}[args.command](args)

if __name__=='__main__':main()
