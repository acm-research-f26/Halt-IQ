"""Prefix-only p(stop), NOT p(correct). Frozen writers/critics are external."""
import math,re,string
from pathlib import Path
import torch
FEATURES={
 'original':['distance','round_fraction','distance_delta','critic_approved','has_previous_draft'],
 'replace':['distance','round_fraction','distance_delta','strands','has_previous_draft'],
 'add':['distance','round_fraction','distance_delta','critic_approved','has_previous_draft','strands'],
}
def normalize(text):
    text=text.lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())
def distance(a,b):
    a,b=set(normalize(a).split()),set(normalize(b).split())
    return 1-len(a&b)/max(1,len(a|b))
def features(rounds_so_far,variant,max_rounds=6):
    if variant not in FEATURES:raise ValueError('Unknown variant')
    n=len(rounds_so_far)
    if max_rounds<1 or not 1<=n<=max_rounds:raise ValueError('Invalid prefix length / round cap')
    for i,r in enumerate(rounds_so_far):
        if not isinstance(r.get('answer'),str):raise ValueError('Each round needs answer text')
        if 'round' in r and r['round']!=i+1:raise ValueError('Provide full contiguous prefix from round one')
    r=rounds_so_far[-1]
    d=distance(rounds_so_far[-2]['answer'],r['answer']) if n>1 else 1.
    delta=d-distance(rounds_so_far[-3]['answer'],rounds_so_far[-2]['answer']) if n>2 else 0.
    if variant in ('original','add'):
        approval=r.get('critic_approved')
        if not isinstance(approval,bool):raise ValueError('critic_approved must be a boolean')
    else:approval=False
    if variant in ('replace','add'):
        score=r.get('strands')
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('strands must be a finite score in [0,1]')
    else:score=0.
    x=[d,n/max_rounds,delta,float(score if variant=='replace' else approval),float(n>1)]
    if variant=='add':x.append(float(score))
    return torch.tensor(x,dtype=torch.float32)
class StopHead(torch.nn.Module):
    def __init__(self,variant='original'):
        super().__init__();self.head=torch.nn.Linear(len(FEATURES[variant]),1)
    def forward(self,x):return self.head(x).squeeze(-1)
class StopPredictor:
    """Load a checkpoint once; call predict(question, rounds_so_far) for p_stop.

    question is retained for interface compatibility; this head does not embed it.
    Labels, gold answers and token totals are NOT prediction inputs.
    Use should_stop() to apply the threshold and mandatory round cap.
    """
    def __init__(self,checkpoint,threshold=.5):
        state=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
        self.variant=state.get('variant','original');self.max_rounds=state.get('max_rounds',6)
        if state['features']!=FEATURES[self.variant]:raise ValueError('Feature schema mismatch')
        if not 0<=threshold<=1:raise ValueError('threshold must be in [0,1]')
        self.threshold=threshold;self.model=StopHead(self.variant)
        self.model.load_state_dict(state['weights']);self.model.eval()
    def predict(self,question,rounds_so_far):
        with torch.no_grad():return torch.sigmoid(self.model(features(rounds_so_far,self.variant,self.max_rounds))).item()
    def should_stop(self,question,rounds_so_far):
        p=self.predict(question,rounds_so_far)
        return len(rounds_so_far)>=self.max_rounds or p>=self.threshold
