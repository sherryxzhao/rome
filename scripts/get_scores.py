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
import csv

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
    if l > 4:
        user = int(sys.argv[4])
        print(f"You have specified user id: {user}") # sherry: 0 jianqi: 1 ryan: 2
    if l > 5:
        batch_idx = int(sys.argv[5])
        print(f"You have specified the batch idx: {batch_idx}") # 10 batch in total
else:
    print("No arguments were passed.")

ftime = open(f'./visualization/runtime-{ds_name}-{user}-{batch_idx}.txt', "a+")
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
log = f"Model and tokenizer initialization finished. {t3-t1}s."
ftime.write(f"{log}\n")
ftime.flush()
print(log)

def read_csv_to_list(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        numbers_list = []
        for row in reader:
            for cell in row:
                if cell:  # This checks if the cell is not empty
                    numbers_list.append(int(cell))  # Convert the cell to an integer
    return numbers_list

def divide_and_extract(lst, N, i):
    """
    Divides a list into N parts and returns the i-th part (0-indexed).
    
    Args:
    lst: The list to divide.
    N: The number of parts to divide the list into.
    i: The index of the part to return (0-indexed).

    Returns:
    A sublist representing the i-th part of the original list.
    """

    # Calculate the size of each part
    part_size = len(lst) // N

    # Calculate the start and end indices of the i-th part
    start_index = i * part_size
    end_index = start_index + part_size

    # Adjust the end index for the last part, if there are remaining elements
    if i == N - 1:
        end_index = len(lst)

    return lst[start_index:end_index]

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

    f = open(f'./visualization/{ds_name}-{user}-{batch_idx}.json', "a+")
    flog = open(f'./visualization/{ds_name}-{user}-{batch_idx}-log.txt', "a+")
    # print the total length of the dataset
    print(f"There are {len(ds)} entries in the dataset")

    # if num is not specified, use total length
    num = len(ds) if num == 0 else num

    print("tracing starts now.")
    t, _t, t5 = time.time(), time.time(), time.time()

    log_freq = 50

    # read in the index list
    index_list = read_csv_to_list("zsre_idx.txt")
    index_list = divide_and_extract(index_list, 3, user)
    index_list = divide_and_extract(index_list, 10, batch_idx)

    cnt = 0 # case counter

    # for dp in tqdm(ds[start:start+num]):
    for i in tqdm(index_list):
        try:
            dp = ds[i]
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
            cnt += 1
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
            # if case_id == start: continue
            if cnt % log_freq == 0:
                log = "Done Processing case_id " +  str(case_id)
                print(log)
                flog.write(f"{log}\n")
                flog.flush()
                _t = time.time()
                log = f"Time used for processing {log_freq} cases: {_t-t}s\n"
                ftime.write(f"{log}\n")
                log = f"Average time for one case: {(_t-t5)/(log_freq)}s\n"
                ftime.write(f"{log}\n")
                ftime.flush()
                t = _t

    f.flush()
    flog.flush()
    f.close()
    flog.close()

# %%
get_scores(ds_name, start, num)

ftime.close()