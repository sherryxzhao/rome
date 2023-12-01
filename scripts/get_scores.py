#%%
import os, re, json
import torch, numpy
import logging
import sys
import time
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR

ftime = open('./visualization/runtime.txt', "a+")
now = datetime.now()
ftime.write(f"Below is time log for run starts at {now}\n")
ftime.flush()

t0 = time.time()
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
    calculate_hidden_flow
)
from dsets import (
    CounterFactDataset,
    KnownsDataset,
    MENDQADataset,
)
t1 = time.time()
log = f"Loading functions finished. {t1-t0}s."
ftime.write(f"{log}\n")
ftime.flush()
print(log)

ds_name = "ZSREeval"
start, num = 0, 0
# Check if any arguments were passed (excluding the script name itself)
l = len(sys.argv)
if l > 1:
    ds_name = sys.argv[1]
    print(f"You have specified dataset name: {ds_name}")
    if l > 2:
        start = int(sys.argv[2])
        print(f"You have specified start entry: {start}")
    if l > 3:
        num = int(sys.argv[3])
        print(f"You have specified number of entries to compute: {num}")
else:
    print("No arguments were passed.")

t2 = time.time()
log = f"Parsing inputs finished. {t2-t1}s."
ftime.write(f"{log}\n")
ftime.flush()
print(log)

torch.set_grad_enabled(False)

IS_COLAB = False
model_name = "gpt2-xl"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=IS_COLAB,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

# Example
# predict_token(
#     mt,
#     ["Megan Rapinoe plays the sport of", "The Space Needle is in the city of"],
#     return_p=True,
# )

t3 = time.time()
log = f"Model and tokenizer initialization finished. {t3-t2}s."
ftime.write(f"{log}\n")
ftime.flush()
print(log)

# %%
def get_scores(ds_name="ZSREeval", start=0, num=0):
    if ds_name == "counterfact":
        ds = CounterFactDataset(DATA_DIR)
    elif ds_name == "ZSREeval":
        ds = MENDQADataset(DATA_DIR, mt.tokenizer)

    t4 = time.time()
    log = f"Loading dataset finished. {t4-t3}s."
    ftime.write(f"{log}\n")
    ftime.flush()
    print(log)

    # print("Computing Noise Level...")
    # noise_level = 3 * collect_embedding_std(mt, [k["requested_rewrite"]["subject"] for k in ds])
    # print(f"Using noise level {noise_level}")

    # use precomputed noise level
    noise_level = 0.13347487896680832
    print(f"Using precomputed noise level: {noise_level}")

    f = open('./visualization/' + ds_name + '_return_mlp.json', "a+")
    flog = open("./visualization/" + ds_name + "_log.txt", "a+")
    # print the total length of the dataset
    print(f"There are {len(ds)} entries in the dataset")

    # if num is not specified, use total length
    num = len(ds) if num == 0 else num

    print("tracing starts now.")
    t, _t, t5 = time.time(), time.time(), time.time()

    log_freq = 100

    for dp in tqdm(ds[start:start+num]):
        try:
            subject = dp["requested_rewrite"]["subject"]
            prompt = dp["requested_rewrite"]["prompt"].replace("{}", subject)
            kind = 'mlp'
            noise = noise_level
            samples = 10
            window = 10
            result = calculate_hidden_flow(
                mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
            )
            scores = result['scores'].cpu().numpy()
            dp["requested_rewrite"]['max_edit_layer'] = int(numpy.argmax(scores[result['subject_range'][1] - 1]))
            # join two dictionary and add to json file
            json_str = json.dumps(dp)
            f.write(json_str+'\n')
            f.flush()


        except Exception as Argument:
            print("Error processing case_id " +  str(dp["case_id"]))
            flog.write("Error processing case_id " +  str(dp["case_id"]))
            flog.write(str(Argument))
            flog.write("\n")
            flog.flush()

        finally:
            case_id = dp["case_id"]
            if (case_id - start) % log_freq == 0:
                log = "Done Processing case_id " +  str(case_id)
                print(log)
                flog.write(f"{log}\n")
                flog.flush()
                _t = time.time()
                log = f"Time used for processing case{case_id-log_freq} to case{case_id}: {_t-t}s\n"
                ftime.write(f"{log}\n")
                log = f"Average time for one case: {(_t-t5)/(case_id+1)}s\n"
                ftime.write(f"{log}\n")
                ftime.flush()
                t = _t

    f.close()
    flog.close()

# %%
get_scores(ds_name, start, num)

ftime.close()