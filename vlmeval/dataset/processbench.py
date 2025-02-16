from .text_base import TextBaseDataset
import numpy as np
from ..smp import *
from datasets import load_dataset
from ..utils import track_progress_rich

def build_dataset():
    # 定义文件名和对应的 JSON 文件
    splits = {
        'gsm8k': 'gsm8k.json',
        'math': 'math.json',
        'olympiadbench': 'olympiadbench.json',
        'omnimath': 'omnimath.json'
    }

    # 初始化一个空的 DataFrame 用于存储所有数据
    all_data = pd.DataFrame()

    # 遍历 splits 中的每个文件
    for category, file_name in splits.items():
        # 构造文件路径
        file_path = f"hf://datasets/Qwen/ProcessBench/{file_name}"
        
        # 加载 JSON 文件
        data = pd.read_json(file_path)
        
        # 添加 category 列
        data['category'] = category
        
        # 将当前文件的数据追加到 all_data DataFrame 中
        all_data = pd.concat([all_data, data], ignore_index=True)
    return all_data



def get_score_PRM(line):
        pred=line['prediction']
        label=line['label']
        if pred == label:
            return 1.0
        else:
            return 0.0

class PROCESSBENCH(TextBaseDataset):
    TYPE = 'PROCESS'

    DATASET_URL = {
        'PROCESSBENCH': ''
    }

    DATASET_MD5 = {'VL-RewardBench': ''}
    data={}

    def __init__(self, dataset='PROCESSBENCH', **kwargs):
        self.dataset_name = dataset

        data = build_dataset()

        data['index'] = [str(x) for x in data['id']]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        problem = line['problem']
        msgs = []

        steps = toliststr(line['steps'])

        msgs = msgs + [dict(type='problem', value=problem)]
        msgs = msgs + [dict(type='steps', value=steps)]
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
            if 'category' in raw_data.columns:
                print("Column 'category' exists in raw_data.")
            else:
                print("Column 'category' does not exist in raw_data.")
            data['prediction'] = [float(x) for x in data['prediction']]
            data['label'] = [x for x in raw_data['label']]
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

        configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
        scores = defaultdict(lambda: 0)

        for config in configs:
            data_this = data[data['category'] == config]
            error_data = data_this[data_this['label'] != -1]
            correct_data = data_this[data_this['label'] == -1]
            
            acc1 = error_data['score'].mean() * 100
            acc2 = correct_data['score'].mean() * 100

            f1 = 2 * acc1 * acc2 / (acc1 + acc2)
            print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')
            scores[config]=f1

        scores['Average']=sum(scores.values()) / len(scores)
        scores = {k: [v] for k, v in scores.items()}

        scores = pd.DataFrame(scores)
        dump(scores, score_file)
        return scores