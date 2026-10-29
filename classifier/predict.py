"""Command-line wrapper for p_stop; consumes observed rounds, never gold labels.

python -m classifier.predict --checkpoint classifier/results/add/seed0/stop_head.pt --input prefix.json
"""
import argparse,json
from pathlib import Path
from .model import StopPredictor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.input.read_text())
    predictor=StopPredictor(a.checkpoint)
    probability=predictor.predict(data.get('question',''),data['rounds'])
    print(json.dumps({'p_stop':probability,'stop':predictor.should_stop(data.get('question',''),data['rounds']),
                      'threshold':predictor.threshold,'max_rounds':predictor.max_rounds,'variant':predictor.variant}))
if __name__=='__main__':main()
