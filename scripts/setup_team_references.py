"""Fetch pinned evaluation dependencies; never alter other team branches."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1] / '.team-reference'
REPO = 'https://github.com/acm-research-f26/Halt-IQ.git'
PINS = {'vedanshi': 'd0688aed897c18d45fea8aacf040896ab06c86ee',
        'Yash': '43b0a61c47845292eee747a01af4a5c87dbd60ab'}
for branch, commit in PINS.items():
    target = ROOT / branch
    if target.exists():
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise SystemExit(f'{target} is at {actual}; expected {commit}. Existing checkout left unchanged.')
        continue
    target.mkdir(parents=True)
    subprocess.run(['git', 'init', str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'remote', 'add', 'origin', REPO], check=True)
    subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1', 'origin', commit], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
print('Pinned shared evaluation dependencies are ready.')
