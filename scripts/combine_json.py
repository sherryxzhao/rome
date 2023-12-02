#%%
import sys, os, json

directory = './ZSRE'
num_ds = 19086
ds_object = []
indices = set()

#%%
def add_to_array(file_path):
    print("Start processing " + str(file_path))
    with open(file_path, "r") as f:
        for line in f:
            print(line)
            json_obj = json.loads(line)
            if json_obj["case_id"] not in indices:
                ds_object.append(json_obj)
                indices.add(json_obj["case_id"])
    print("Done processing " + str(file_path))

def dump_json(file_path):
    sorted_ds_object = sorted(ds_object, key=lambda x: x['case_id'])
    with open(file_path, "w+") as f:
        for obj in sorted_ds_object:
            f.write(json.dumps(obj) + '\n')
    print("Done writing generated json file to " + file_path)

def remaining(file_path):
    count = 0
    with open(file_path, "w+") as f:
        for i in range(num_ds):
            if ((i + 1) not in indices):
                f.write(str(i + 1) + ',')
                count += 1
    print("Done writing remaining case_id to " + file_path)
    print("There are " + str(count) + " data points in total")

#%%
# iterate over files
for filename in os.scandir(directory):
    if filename.is_file() and filename.name.endswith('.json'):
        add_to_array(filename)

#%%
dump_json('./dsets/zsre_comb.json')
#%%
remaining('./visualization/zsre_idx.txt')