#!/bin/zsh
# Steps 5-6: logprob critic on `extra`, then qwen3:14b on `subset80`.
# One Ollama job at a time; every model job is killed at 4:00 pm. Scores are cached
# per draft, so stopping loses nothing. Reports (no model calls) run at the end.
cd "$(dirname "$0")"
DEADLINE=$(date -j -f "%H:%M" "16:00" +%s)

stage() {  # run a command, killing it at the deadline
  if [ $(date +%s) -ge $DEADLINE ]; then echo "SKIP (past 4:00 pm): $*"; return; fi
  echo "START $(date +%H:%M): $*"
  "$@" & local pid=$!
  ( sleep $((DEADLINE - $(date +%s))); kill $pid 2>/dev/null && echo "STOPPED at 4:00 pm: $*" ) & local watchdog=$!
  wait $pid
  kill $watchdog 2>/dev/null
  echo "DONE  $(date +%H:%M): $*"
}

stage python3 score_drafts.py qwen8b-logprob extra
# Unload qwen3:8b so qwen3:14b has the memory (24 GB Mac).
curl -s http://localhost:11434/api/generate -d '{"model": "qwen3:8b", "keep_alive": 0}' > /dev/null
echo "loaded models after unload: $(curl -s http://localhost:11434/api/ps)"
stage python3 score_drafts.py qwen14b subset80 --limit 3   # timing check
stage python3 score_drafts.py qwen14b subset80
python3 report.py > /dev/null && python3 report.py --label lenient > /dev/null && python3 combos.py > /dev/null
echo "ALL FINISHED $(date +%H:%M)"
