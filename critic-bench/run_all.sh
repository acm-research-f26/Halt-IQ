#!/bin/zsh
# Run the remaining scoring stages in order, one Ollama job at a time, stopping at 3:30 pm.
# Every stage caches per draft, so stopping (or rerunning this script) loses nothing.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
DEADLINE=$(date -j -f "%H:%M" "15:30" +%s)

while pgrep -f "python3 make_extra.py write" > /dev/null; do sleep 10; done  # wait for the writer

stage() {  # run a command, killing it at the deadline
  if [ $(date +%s) -ge $DEADLINE ]; then echo "SKIP (past 3:30 pm): $*"; return; fi
  echo "START $(date +%H:%M): $*"
  "$@" & local pid=$!
  ( sleep $((DEADLINE - $(date +%s))); kill $pid 2>/dev/null && echo "STOPPED at 3:30 pm: $*" ) & local watchdog=$!
  wait $pid
  kill $watchdog 2>/dev/null
  echo "DONE  $(date +%H:%M): $*"
}

for set in first all extra; do stage python3 score_drafts.py evidence_match $set; done
stage python3 make_samples.py                      # Ollama: consistency samples
for set in first extra; do stage python3 score_drafts.py consistency $set; done
stage python3 score_drafts.py laya extra
stage python3 score_drafts.py kev extra
stage python3 score_drafts.py llm extra            # Ollama: llm critic
echo "ALL STAGES FINISHED $(date +%H:%M)"
