from .text_base import TextBaseDataset
import numpy as np
from ..smp import *
from datasets import load_dataset
from ..utils import track_progress_rich
import pandas as pd

# 定义一个函数，根据 subset 的值返回 category 的值
def get_category(subset):
    if subset in ['mt-bench-easy', 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-med', 'alpacaeval-easy']:
        return 'Chat'
    elif subset in ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-GPTOut', 'llmbar-adver-neighbor', 
                    'llmbar-adver-GPTInst', 'llmbar-adver-manual']:
        return 'Chat_Hard'
    elif subset in ['refusals-offensive', 'refusals-dangerous', 'donotanswer', 'xstest-should-respond', 
                    'xstest-should-refuse']:
        return 'Safety'
    elif subset in ['hep-cpp', 'hep-go', 'hep-js', 'hep-rust', 'hep-python', 'math-prm', 'hep-java']:
        return 'Reasoning'
    else:
        return 'Unknown'  # 如果 subset 不在上述列表中，返回 'Unknown'



def build_dataset():
    splits = {'raw': 'data/raw-00000-of-00001.parquet', 'filtered': 'data/filtered-00000-of-00001.parquet'}
    data = pd.read_parquet("hf://datasets/allenai/reward-bench/" + splits["filtered"])
  
    data['category'] = data['subset'].apply(get_category)
    
    #extra_data=pd.read_parquet("hf://datasets/allenai/pref-test-sets/")
    #extra_data['category']=['Prior_Sets'] * len(extra_data)
    #all_data = pd.concat([data, extra_data], ignore_index=True)
    return data



def get_score_PRM(line):
        pred=line['prediction']
        label=line['label']
        if pred == label:
            return 1.0
        else:
            return 0.0

class RewardBench(TextBaseDataset):
    TYPE = 'PREFERENCE'

    DATASET_URL = {
        'RewardBench': ''
    }

    DATASET_MD5 = {'RewardBench': ''}
    data={}

    def __init__(self, dataset='RewardBench', **kwargs):
        self.dataset_name = dataset

        data = build_dataset()

        data['index'] = [str(x) for x in data['id']]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        prompt = line['prompt']
        msgs = []

        chosen = line['chosen']
        rejected = line['rejected']

        response = [chosen, rejected]

        msgs = msgs + [dict(type='prompt', value=prompt)]
        msgs = msgs + [dict(type='response', value=response)]
        return msgs

    @classmethod
    def evaluate_PRM(self, eval_file):
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', '_evaluate.xlsx')
        score_file = eval_file.replace(f'.{suffix}', '_score.csv')
        tmp_file = eval_file.replace(f'.{suffix}', '_evaluate.pkl')

        if not osp.exists(storage):
            raw_data = build_dataset()
            data = load(eval_file)
            data['prediction'] = [float(x) for x in data['prediction']]
            data['label'] = [1] * len(raw_data)
            data['category'] = [x for x in raw_data['category']]
            

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    get_score_PRM,
                    tups,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    ans[k] = v

            data['score'] = [ans[idx] for idx in data['index']]
            # data.pop('image')
            dump(data, storage)

        data = load(storage)
        lt = len(data)

        category_scores = defaultdict(lambda: 0)
        category_cnt = defaultdict(lambda: 0)
        scores = defaultdict(lambda: 0)
        for i in range(lt):
            item = data.iloc[i]
            category_scores[item['category']] += item['score']
            category_cnt[item['category']] += 1
        # calculate the average score for each category
        for k, v in category_scores.items():
            scores[k] = v / category_cnt[k]
        # calculate category macro accuracy (average across categories)
        scores['Score'] = sum(scores.values()) / 4

        scores = {k: [v] for k, v in scores.items()}

        scores = pd.DataFrame(scores)
        dump(scores, score_file)
        return scores