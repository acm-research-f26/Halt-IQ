# Audited stopping-policy comparison

Source: live; 60 test questions; seed 1. Cached replay.

|Policy|F1|EM|Rounds|Tokens|Calls|Saved vs Fixed 6|
|---|---:|---:|---:|---:|---:|---:|
|fixed_1|0.4626|0.3500|1.00|1241.9|2.00|83.44%|
|fixed_3|0.4828|0.3667|3.00|3745.1|6.00|50.06%|
|fixed_6|0.4828|0.3667|6.00|7499.0|12.00|0.00%|
|critic|0.4626|0.3500|1.08|1344.4|2.17|82.07%|
|distance|0.4828|0.3667|3.07|3824.7|6.13|49.00%|
|learned|0.4626|0.3500|1.00|1241.9|2.00|83.44%|
|oracle_f1|0.4995|0.3833|1.05|1306.3|2.10|82.58%|
