# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
import os
from collections import defaultdict
import json
import torch
import random
import numpy as np
import pdb

LMs = [
    # {
    #     "lm": "transformerxl",
    #     "label": "transformerxl",
    #     "models_names": ["transformerxl"],
    #     "transformerxl_model_name": "transfo-xl-wt103",
    #     "transformerxl_model_dir": "pre-trained_language_models/transformerxl/transfo-xl-wt103/",
    # },
    # {
    #     "lm": "elmo",
    #     "label": "elmo",
    #     "models_names": ["elmo"],
    #     "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
    #     "elmo_vocab_name": "vocab-2016-09-10.txt",
    #     "elmo_model_dir": "pre-trained_language_models/elmo/original",
    #     "elmo_warm_up_cycles": 10,
    # },
    # {
    #     "lm": "elmo",
    #     "label": "elmo5B",
    #     "models_names": ["elmo"],
    #     "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
    #     "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
    #     "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
    #     "elmo_warm_up_cycles": 10,
    # },
    {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
    },
    # {
    #     "lm": "bert",
    #     "label": "bert_large",
    #     "models_names": ["bert"],
    #     "bert_model_name": "bert-large-cased",
    #     "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    # },
]


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")
    uid_list_all, mask_feature_list_all, answers_list_all = [], [], []
    all_correct_uuids = []
    total_modified_correct, total_unmodified_correct = 0, 0
    total_modified_num, total_unmodified_num = 0, 0
    for relation in relations:
        # if "type" not in relation or relation["type"] != "1-1":
        #     continue

        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": 'pre-trained_language_models/bert/cased_L-12_H-768_A-12/vocab.txt',#"pre-trained_language_models/common_vocab_cased.txt",
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": 512, # change to 512 later
            "threads": 2,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
            "return_features": False,
            "uuid_list": []
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        if getattr(args, 'output_feature_path', ''):
            # Get the features for kNN-LM. Ignore this part if only obtaining the correct-predicted questions.
            Precision1, total_unmodified, Precision1_modified, total_modified, uid_list, mask_feature_list, answers_list = run_evaluation(args, shuffle_data=False, model=model)
            if len(uid_list) > 0:
                uid_list_all.extend(uid_list)
                mask_feature_tensor = torch.cat(mask_feature_list, dim=0)
                mask_feature_list_all.append(mask_feature_tensor)
                answers_list_all.extend(answers_list)

        else:
            Precision1, total_unmodified, Precision1_modified, total_modified, correct_uuids = run_evaluation(args, shuffle_data=False, model=model)
            all_correct_uuids.extend(correct_uuids)

        total_modified_correct += Precision1_modified
        total_unmodified_correct += Precision1
        total_modified_num += total_modified
        total_unmodified_num += total_unmodified
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    print("Unmodified acc: {}, modified acc: {}".format(total_unmodified_correct / float(total_unmodified_num),
                            0 if total_modified_num == 0 else total_modified_correct / float(total_modified_num)))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )
    if len(uid_list_all) > 0:
        out_dict = {'mask_features': torch.cat(mask_feature_list_all, dim=0),
                    'uuids': uid_list_all,
                    'obj_labels': answers_list_all}
        torch.save(out_dict, 'datastore/ds_change32.pt')
    if len(all_correct_uuids) > 0:
        if not os.path.exists('modification'):
            os.makedirs('modification')
        json.dump(all_correct_uuids, open('modification/correct_uuids.json', 'w'))
    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file(os.path.join(data_path_pre, 'relations.jsonl'))
    data_path_pre = os.path.join(data_path_pre, "TREx/")
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)


if __name__ == "__main__":
    lama_data_path = 'lama_data/data'

    print("1. Getting the correctly predicted uuids and save them to modification/correct_uuids.json.")
    parameters = get_TREx_parameters(lama_data_path)
    run_all_LMs(parameters)

    print("2. Sample the correctly predicted questions and replace their answers.")
    num_modified_facts = 32
    uuid_list = json.load(open('modification/correct_uuids.json'))

    # Sample the correctly predicted questions.
    random.shuffle(uuid_list)
    selected_uids = uuid_list[:num_modified_facts]
    json.dump(selected_uids, open(f'modification/change_list_{num_modified_facts}.json', 'w'))

    # Select the replaced answers according to the frequency.
    # First, construct the mappings from relations to objects
    uid2rel_dict, rel2objs = {}, defaultdict(list)
    uid2obj_dict = {}
    obj_freq_dict = defaultdict(int)
    trex_data_path = os.path.join(lama_data_path, 'TREx')
    files = os.listdir(trex_data_path)
    for file in files:
        fullpath = os.path.join(trex_data_path, file)
        for line in open(fullpath):
            data_dict = json.loads(line)
            uuid = data_dict['uuid']
            obj = data_dict['obj_label']
            rel = data_dict['predicate_id']
            uid2rel_dict[uuid] = rel
            uid2obj_dict[uuid] = obj
            if obj not in rel2objs[rel]:
                rel2objs[rel].append(obj)
            obj_freq_dict[obj] += 1

    uuid_list = json.load(open('modification/change_list_32.json'))
    uuid2newans_dict = {}
    for uuid in uuid_list:
        rel = uid2rel_dict[uuid]
        all_objs = rel2objs[rel]
        obj_freqs = [obj_freq_dict[o] for o in all_objs]
        orig_obj = uid2obj_dict[uuid]
        obj_freqs = np.array([f * float(o != orig_obj) for f, o in zip(obj_freqs, all_objs)])
        rand_obj = np.random.choice(all_objs, p=obj_freqs / np.sum(obj_freqs))
        uuid2newans_dict[uuid] = rand_obj

    json.dump(uuid2newans_dict, open(f'modification/change_list_{num_modified_facts}.json', 'w'))
