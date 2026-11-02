# Yash - November 2 orchestration results

[Professor results report](Yash-Nov2-Results-Report-20261102.pdf) | [Plain text](Yash-Nov2-Results-Report-20261102.txt)

This report presents the same eleven HotpotQA questions across seven approaches. It leads with normalized exact-answer accuracy, answer-token F1, measured role calls and the implemented writer/reviewer/challenger flow. All 77 question-arm attempts are included, with eight failed attempts scored zero.

The Strands-informed linear policy answered 6/11 exactly (54.5%), with answer F1 59.1 on a 0-100 scale. The verifier and inherited AND loops answered 4/11 (36.4%), with F1 40.9. Measured role calls were 22 versus 54 for each loop. One writer also answered 6/11 exactly, with the highest F1 (65.2) and the fewest calls (11).

- [All twelve metrics and recorded resources](Yash-Nov2-Observed11-Metrics-20261102.csv)
- [Accuracy/call figure](Yash-Nov2-Observed11-Accuracy-Calls-20261102.png) and [vector figure](Yash-Nov2-Observed11-Accuracy-Calls-20261102.svg)
- [Separate paper summary](Yash-Nov2-Separate-Paper-Summary-20261102.txt)
- [Source/product manifest](Yash-Nov2-Results-Manifest-20261102.json) and [verification](Yash-Nov2-Results-Verification-20261102.json)

Author: Yash Baruah. Vedanshi and Shree are credited for the baseline methods. The report is a concise presentation of observed results; it adds no inference or training and changes no scores or frozen study plan. Original evidence and earlier reporting remain preserved separately.
