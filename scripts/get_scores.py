
#%%
import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
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

torch.set_grad_enabled(False)

IS_COLAB = False
model_name = "gpt2-xl"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=IS_COLAB,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

# %%
predict_token(
    mt,
    ["Megan Rapinoe plays the sport of", "The Space Needle is in the city of"],
    return_p=True,
)

# %%
def get_scores(ds_name):
    if ds_name == "counterfact":
        ds = CounterFactDataset(DATA_DIR)
    elif ds_name == "ZSREeval":
        ds = MENDQADataset(DATA_DIR, mt.tokenizer)

    print("Computing Noise Level...")
    noise_level = 3 * collect_embedding_std(mt, [k["requested_rewrite"]["subject"] for k in ds])
    print(f"Using noise level {noise_level}")

    f = open('./visualization/' + ds_name + '_return_mlp.json', "a+")

    # for each entry in the dataset
    for dp in ds[:]:
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
            dp["requested_rewrite"]['max_edit_layer'] = int(numpy.argmax(scores[result['subject_range'][1]]))
            # join two dictionary and add to json file
            json_str = json.dumps(dp)
            f.write(json_str+'\n')
        except:
            print("Error processing case_id " +  dp["case_id"])
    f.close()

# %%
get_scores("ZSREeval")
