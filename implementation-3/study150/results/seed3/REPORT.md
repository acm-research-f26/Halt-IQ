# Audited stopping-policy comparison

Source: live; 80 test questions; seed 3. Cached replay.

|Policy|F1|EM|Rounds|Tokens|Calls|Saved vs Fixed 6|
|---|---:|---:|---:|---:|---:|---:|
|fixed_1|0.3588|0.2125|1.00|3027.7|2.00|83.39%|
|fixed_3|0.3824|0.2250|3.00|9107.7|6.00|50.03%|
|fixed_6|0.3921|0.2375|6.00|18227.4|12.00|0.00%|
|critic|0.3588|0.2125|1.12|3320.5|2.25|81.78%|
|distance|0.3921|0.2375|3.14|9544.3|6.28|47.64%|
|learned|0.3588|0.2125|1.00|3027.7|2.00|83.39%|
|oracle_f1|0.3921|0.2375|1.06|3203.2|2.12|82.43%|
