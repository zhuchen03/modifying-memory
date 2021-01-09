from collections import defaultdict
import json
import os
import numpy as np
import random
import copy
import pdb

if __name__ == '__main__':
    data_path = 'zsre_data/structured_zeroshot-train-kilt.jsonl'
    num_modified_facts = 32

    print("1. Split the original training set of zsRE into training and eval sets.")
    train_split, eval_split = [], []
    num_train, num_eval = 0, 0
    train_id2idx, eval_id2idx = {}, {} # used to store valid ids
    answer_freq, rel2answer = defaultdict(int), defaultdict(list)
    with open(data_path, 'r') as fs:
        for line in fs:
            datum = json.loads(line)
            id = datum['id']
            answer = datum['output'][0]['answer']
            questions = datum['meta']['template_questions']
            rel = datum['input'].split('[SEP] ')[1]
            if answer not in rel2answer[rel]:
                rel2answer[rel].append(answer)
            answer_freq[answer] += 1

            if len(questions) == 1:
                train_idxes = [0]
                eval_idxes = []
            elif len(questions) == 2:
                train_idxes = [np.random.choice([0, 1])]
                eval_idxes = [1 - train_idxes[0]]
                train_id2idx[id] = len(train_split)
                eval_id2idx[id] = len(eval_split)
            else:
                # always use 2 questions for evaluation
                eval_idxes = np.random.choice([i for i in range(len(questions))], size=2, replace=False)
                train_idxes = [i for i in range(len(questions)) if i not in eval_idxes]
                train_id2idx[id] = len(train_split)
                eval_id2idx[id] = len(eval_split)

            masked_questions = []
            for idx in train_idxes:
                masked_questions.append(questions[idx] + ' [MASK]')
            num_train += len(train_idxes)
            train_split.append({'id': id, 'answer': answer, 'questions': masked_questions, 'rel': rel})

            if len(eval_idxes) > 0:
                masked_questions = []
                for idx in eval_idxes:
                    masked_questions.append(questions[idx] + ' [MASK]')
                num_eval += len(masked_questions)
                eval_split.append({'id': id, 'answer': answer, 'questions': masked_questions, 'rel': rel})
        print('Number of training facts: {}, eval facts: {}, training questions: {}, eval questions: {}'.format(
            len(train_split), len(eval_split), num_train, num_eval
        ))

    print("\n\n2. Saving unmodified data... ")
    dst_path = 'modification'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    with open(os.path.join(dst_path, 'zsre_train_unmodified.jsonl'), 'w') as fs:
        for datum in train_split:
            out_str = json.dumps(datum)
            fs.write(out_str)

    with open(os.path.join(dst_path, 'zsre_eval_unmodified.jsonl'), 'w') as fs:
        for datum in eval_split:
            out_str = json.dumps(datum)
            fs.write(out_str)

    print("\n\n3. Generating and saving modified data... ")
    id_list = list(train_id2idx.keys())
    random.shuffle(id_list)
    selected_uids = id_list[:num_modified_facts]
    modified_train, modified_eval = [], []
    for id in selected_uids:
        train_datum = train_split[train_id2idx[id]]
        eval_datum = eval_split[eval_id2idx[id]]

        orig_ans = train_datum['answer']
        rel = train_datum['rel']
        all_ans = rel2answer[rel]
        ans_freq = np.array([answer_freq[ans] * float(ans != orig_ans) for ans in all_ans])
        new_ans = np.random.choice(all_ans, p=ans_freq / np.sum(ans_freq))

        m_train = copy.deepcopy(train_datum)
        m_train['answer'] = new_ans
        m_eval = copy.deepcopy(eval_datum)
        m_eval['answer'] = new_ans
        modified_train.append(m_train)
        modified_eval.append(m_eval)

    with open(os.path.join(dst_path, f'zsre_train_modified_{num_modified_facts}.jsonl'), 'w') as fs:
        for datum in modified_train:
            out_str = json.dumps(datum)
            fs.write(out_str)

    with open(os.path.join(dst_path, f'zsre_eval_modified_{num_modified_facts}.jsonl'), 'w') as fs:
        for datum in modified_eval:
            out_str = json.dumps(datum)
            fs.write(out_str)
