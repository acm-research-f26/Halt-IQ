"""Faithful original policy-gradient loop, three feature variants, five seeds."""
import argparse,json,random,hashlib
from pathlib import Path
import torch
from .model import StopHead,FEATURES,features
ROOT=Path(__file__).resolve().parent
CONFIG={'epochs':20,'group_size':8,'cost':.1,'learning_rate':.01,'max_rounds':6,'threshold':.5,'seeds':list(range(5)),
        'selection':'strictly highest validation reward; keep earliest tie',
        'reward':'answer_f1 - 0.1 * original_Qwen_prefix_tokens / original_Qwen_six_round_tokens',
        'algorithm':'original single-update group-relative policy gradient, no KL; not full GRPO'}
def rollout(model,rounds,variant,greedy=False):
    decisions=[]
    for t in range(len(rounds)-1):
        x=features(rounds[:t+1],variant,CONFIG['max_rounds'])
        dist=torch.distributions.Bernoulli(logits=model(x))
        action=(dist.probs>=.5).float() if greedy else dist.sample()
        decisions.append((x,action,dist.log_prob(action).detach()))
        if action.item():return t,decisions
    return len(rounds)-1,decisions

def reward(rounds,index):
    return rounds[index]['quality']-.1*sum(r['tokens'] for r in rounds[:index+1])/max(1,sum(r['tokens'] for r in rounds))

def train_one(items,variant,seed,out):
    training=[x for x in items if x['split']=='train'];dev=[x for x in items if x['split']=='dev']
    assert len(training)==50 and len(dev)==20
    torch.set_num_threads(1);torch.manual_seed(seed);rng=random.Random(seed)
    model=StopHead(variant);opt=torch.optim.Adam(model.parameters(),lr=.01)
    best=-float('inf');history=[];out.mkdir(parents=True,exist_ok=True)
    for epoch in range(20):
        rng.shuffle(training);signals=0
        for item in training:
            rounds=item['rounds']
            with torch.no_grad():group=[rollout(model,rounds,variant) for _ in range(8)]
            returns=torch.tensor([reward(rounds,t) for t,_ in group])
            if returns.std(unbiased=False)<1e-8:continue
            advantages=(returns-returns.mean())/(returns.std(unbiased=False)+1e-6)
            losses=[]
            for (_,decisions),adv in zip(group,advantages):
                if not decisions:continue
                current=sum(torch.distributions.Bernoulli(logits=model(x)).log_prob(a) for x,a,_ in decisions)
                old=sum(lp for _,_,lp in decisions);ratio=torch.exp(current-old)
                losses.append(-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv))
            if losses:
                opt.zero_grad();torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();signals+=1
        with torch.no_grad():score=sum(reward(x['rounds'],rollout(model,x['rounds'],variant,True)[0]) for x in dev)/len(dev)
        history.append({'epoch':epoch+1,'dev_reward':score,'groups_with_signal':signals})
        if score>best:
            best=score
            torch.save({'weights':model.state_dict(),'features':FEATURES[variant],'variant':variant,'seed':seed,
                        'max_rounds':6,'threshold':.5,'selected_epoch':epoch+1,'dev_reward':score,'config':CONFIG,
                        'train_ids':[x['id'] for x in training],'dev_ids':[x['id'] for x in dev],
                        'data_sha256':hashlib.sha256((ROOT/'inputs/trajectories.json').read_bytes()).hexdigest()},out/'stop_head.pt')
    (out/'training.json').write_text(json.dumps({'variant':variant,'seed':seed,'history':history},indent=2)+'\n')
    print(f'{variant} seed={seed} best_dev_reward={best:.6f}',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results');args=p.parse_args()
    items=json.loads((ROOT/'inputs/trajectories.json').read_text())
    args.output.mkdir(exist_ok=True,parents=True)
    (args.output/'training_config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    # All 15 training/validation runs finish before evaluate.py reads test results.
    for variant in FEATURES:
        for seed in CONFIG['seeds']:train_one(items,variant,seed,args.output/variant/f'seed{seed}')
if __name__=='__main__':main()
