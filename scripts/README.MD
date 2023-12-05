## Eval seperate tasks
- copy `eval_seperate_tasks.py` and `run_eval_sbatches.sh` to the head node.
- run `eval_seperate_tasks.py {user_id}`, user_id is `1`, `2`, or `3`.
- Each user will run around `6000` cases, seperated into tasks with each containing ~ `1000` cases.
- Remove results in `scratch/rome/results/ROME/run_eval*` if you have run it before because we are using run from continue.
- Execute `run_eval_sbatches.sh`.