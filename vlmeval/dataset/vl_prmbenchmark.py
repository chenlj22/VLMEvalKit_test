from ast import literal_eval

from .image_base import ImageBaseDataset, img_root_map
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


LLM_PARSE_ANSWER_PROMPT = '''
You are given a stepwise judgement of an answer. It will be formatted as follows:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>

...

<analysis_n>
...(analysis of paragraph n)...
</analysis_n>

<conclusion>
Correct/Incorrect
</conclusion>

Please summarize the judgement of each paragraph in the format of a list, in which -1 means Incorrect, and 1 means Correct.
Return the list ONLY. \
    e.g., [-1,1,1,-1]
*The length of the list should be equal to {num_steps}.\ 
    e.g., The list [1, -1, 1, 1] corresponds to 4.
    
--------------------------------------------------

The following is the judgement for your task:
[Judgement] 
{judgement}
'''

INSTRUCTION = """
I will provide a math problem along with a solution. They will be formatted as 
follows:

[Math Problem]

<math_problem>
...(math problem)...
</math_problem>

[Solution]

<paragraph_1>
...(paragraph 1 of solution)...
</paragraph_1>

...

<paragraph_n>
...(paragraph n of solution)...
</paragraph_n>

Your task is to review each paragraph of the solution in sequence, analyzing, 
verifying, and critiquing the reasoning in detail. You need to provide the 
analyses and the conclusion in the following format:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>

...

<analysis_n>
...(analysis of paragraph n)...
</analysis_n>

<conclusion>
Correct/Incorrect
</conclusion>


* When you analyze each paragraph, you should use proper verification, 
recalculation, or reflection to indicate whether it is logically and 
mathematically valid. Please elaborate on the analysis process carefully.

* If an error is detected in any paragraph, you should describe the nature and 
cause of the error in detail, and suggest how to correct the error or the correct 
approach. If an error is found in any step, mark that step as incorrect but 
continue analyzing the remaining steps to determine their correctness. 
rovide a detailed conclusion for each step.

For instance, given a solution of five paragraphs, if an error is found in the 
third paragraph and another error is found in the fifth paragraph, you should reply in the following format:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>

<analysis_2>
...(analysis of paragraph 2)...
</analysis_2>

<analysis_3>
...(analysis of paragraph 3; since an error is found here, also provide detailed 
critique and correction guideline)...
</analysis_3>

<analysis_4>
...(analysis of paragraph 4)...
</analysis_4>

<analysis_5>
...(analysis of paragraph 5; since an error is found here, also provide detailed 
critique and correction guideline)...
</analysis_5>

...

<conclusion>
Incorrect
</conclusion>

* Respond with your analyses and conclusion directly.

--------------------------------------------------

The following is the math problem and the solution for your task:

[Math Problem]

{query}

[Solution]

{tagged_response}
""".strip()

PROMPT_TEMPLATE = '''\
You are a highly capable multimodal AI assistant, tasked with evaluating the correctness of the multi-step processes involved in answering visual questions.
Please analyze the following image and question, then determine the correctness of the steps.

Question: {query}

Steps:
{steps_list}

Please evaluate the steps of the answer based on the following criteria:
1. Coherence: Assess whether each step logically follows from the previous one, ensuring a smooth and connected sequence of actions or reasoning.
2. Logic: Determine if the steps follow a clear and rational order, with each step contributing to a coherent overall process.
3. Accuracy: Evaluate whether each step accurately addresses the question and aligns with the visual information provided in the image.
4. Brevity: Check if each step is concise and to the point, avoiding unnecessary details while providing all essential information.


During your evaluation, please:
1. Ensure that you thoroughly examine each step to determine if any are missing or if any additional steps are needed to fully address the question. 
2. Your evaluation should cover the completeness and clarity of the steps as well.
3. The number of the steps you evaluated should not be bigger than the number of the steps the answer provided.

After your evaluation, please:
1. Explain your reasoning for each criterion.
2. Provide a stepwise judgment on each step(Correct or Wrong).
3. Your answer should be in a format like this example:\
Step1: 
    Stepwise judgement: Correct; 
    Reason: Your reason.
Step2: 
    Stepwise judgement: Wrong; 
    Reason: Your reason.

Your response should be structured and detailed, \
demonstrating your understanding of both the visual and textual elements of the task.
'''

def format_steps(steps):
    """
    格式化步骤列表，每个步骤前后分别标注 <paragraph_n> 和 </paragraph_n>。
    """
    formatted_steps = "\n".join(f"<paragraph_{i+1}>\n{step}\n</paragraph_{i+1}>" for i, step in enumerate(steps))
    return formatted_steps



def get_score_PRM(model, line):
    pred=VL_PRMBenchmark_eval_answer(model, line)
    human_ranking = line['human_ranking']
    # pred=line['prediction']
    # 初始化计数器
    count = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    pred = literal_eval(pred)
    # 确保两个列表长度相同
    if len(human_ranking) != len(pred):
        # 获取较短列表的名称和长度
        shorter_list_name = "human_ranking" if len(human_ranking) < len(pred) else "pred"
        shorter_list_len = min(len(human_ranking), len(pred))

        # 裁剪较长的列表
        if len(human_ranking) > len(pred):
            human_ranking = human_ranking[:shorter_list_len]
        elif len(human_ranking) < len(pred):
            pred = pred[:shorter_list_len]

        # 打印警告信息
        print(f"Warning: The length of human_ranking and pred is different. "
            f"It has been aligned to the length of the shorter list '{shorter_list_name}' ({shorter_list_len}).")

    # 遍历两个列表，统计每个指标的次数
    for hr, p in zip(human_ranking, pred):
        if hr == 1 and p == 1:
            count['TP'] += 1  # True Positive
        elif hr == -1 and p == -1:
            count['TN'] += 1  # True Negative
        elif hr == -1 and p == 1:
            count['FP'] += 1  # False Positive
        elif hr == 1 and p == -1:
            count['FN'] += 1  # False Negative

    return count

def VL_PRMBenchmark_eval_answer(model, line):
    steps = literal_eval(line['response'])['steps']
    prompt = LLM_PARSE_ANSWER_PROMPT.format(judgement=line['prediction'], num_steps=len(steps))
    messages = [dict(type='text', value=prompt)]

    resp = model.generate(messages)
    print(resp)
    print(type(resp))
    print('--------------')
    if resp is None:
        return 'Unknown'
    
    try:
        # 尝试将 resp 转换为列表
        eval_resp = eval(resp)
        # 检查转换后的结果是否为列表
        if isinstance(eval_resp, list):
            pass
        else:
            # 如果不是列表，替换为仅含一个元素 0 的列表
            resp = str([0])
    except (SyntaxError, NameError):
        # 如果 resp 不是有效的列表表示，替换为仅含一个元素 0 的列表
        resp = str([0])

    # 检查字符串是否以右方括号 ] 结尾
    if not resp.endswith(']'):
        # 如果不以右方括号结尾，则补全
        resp = resp.rstrip(']') + ']'

    print(resp)
    print('------------------------------')

    return resp


class VL_PRMBenchmark(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'VL_PRMBenchmark': ''
    }
    DATASET_MD5 = {'VL_PRMBenchmark': ''}
    DATA_PATH = '/mnt/petrelfs/share_data/wangweiyun/share_wwy/annotation/250207/250207-merged.jsonl'
    IMAGE_PATH = '/mnt/petrelfs/share_data/wangweiyun/share_wwy/annotation/250207'
    from_internal = True

    def __init__(self, dataset='VL_PRMBenchmark', skip_noimg=True):
        ROOT = LMUDataRoot()
        self.dataset_name = dataset

        if self.from_internal:
            # 如果 from_internal 为 True，从内部路径加载数据集
            data = self.load_data_from_internal(self.DATA_PATH)
            self.img_root = self.IMAGE_PATH
        else:
            # 如果 from_internal 为 False，使用父类的初始化逻辑
            self.img_root = osp.join(LMUDataRoot(), 'images', img_root_map(dataset))
            data = self.load_data(dataset)

        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]

        data['index'] = range(1, len(data) + 1)

        data['index'] = [str(x) for x in data['index']]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data

        self.process_image_paths()

        self.post_build(dataset)

    def load_data_from_internal(self, data_path: str) -> pd.DataFrame:
        """
        从内部路径加载数据集文件（jsonl 格式）。
        """
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        return df

    def process_image_paths(self):
        """
        处理数据集中的 'image' 字段，将其转换为完整的图像路径。
        """
        if 'image' in self.data:
            self.data['image_path'] = self.data['image'].apply(
                lambda x: [os.path.join(self.IMAGE_PATH, img_path) for img_path in toliststr(x)]
            )
            self.data.drop(columns=['image'], inplace=True)

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)  # save image to local
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        steps = line['response']['steps']
        image_path=line['image_path']
        prompt = PROMPT_TEMPLATE.format(
            query=question, steps_list=format_steps(steps), 
        )
        msgs = msgs + [dict(type='text', value=prompt)]
        msgs = msgs + [dict(type='question', value=question)]
        msgs = msgs + [dict(type='steps', value=steps)]
        msgs = msgs + [dict(type='image_path', value=image_path)]
        return msgs

    def build_prompt_vlm(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)  # save image to local
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        steps = line['response']['steps']
        image_path=line['image_path']
        # prompt = PROMPT_TEMPLATE.format(
        #     query=question, steps_list=format_steps(steps), num_steps=len(steps),
        # )
        prompt = INSTRUCTION.format(
            query=question, tagged_response=format_steps(steps)
        )
        msgs = msgs + [dict(type='text', value=prompt)]
        msgs += [{'type': 'image', 'value': img_path} for img_path in image_path]
        return msgs
        

    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.csv')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            raw_data = VL_PRMBenchmark('VL_PRMBenchmark').data
            data = load(eval_file)
            data['prediction'] = [str(x) for x in data['prediction']]
            data['human_ranking'] = raw_data['response'].apply(lambda x: x['process_correctness'])

            
            judge_kwargs['temperature'] = 0
            judge_kwargs['timeout'] = 60
            model = build_judge(max_tokens=128, **judge_kwargs)

            assert model.working(), (
                'VL_PRMBenchmark evaluation requires a working OPENAI API\n'
                + DEBUG_MESSAGE
            )

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    get_score_PRM,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    ans[k] = v

            data['count_line'] = [ans[idx] for idx in data['index']]
            # data.pop('image')
            dump(data, storage)

        data = load(storage)
        lt = len(data)

        category_scores = defaultdict(lambda: 0)
        category_cnt = defaultdict(lambda: 0)
        scores = defaultdict(lambda: 0)

        '''for i in range(lt):
            item = data.iloc[i]
            category_scores[item['category']] += item['score']
            category_cnt[item['category']] += 1
        # calculate the average score for each category
        for k, v in category_scores.items():
            scores[k] = v / category_cnt[k]
        # calculate category macro accuracy (average across categories)
        scores['Macro Accuracy'] = sum(scores.values()) / len(scores)
        # calculate the total average score
        scores['Overall Consistency'] = sum(category_scores.values()) / lt'''

        TP=0
        TN=0
        FP=0
        FN=0

        for i in range(lt):
            item = data.iloc[i]
            count_line_dict = json.loads(item['count_line'].replace("'", '"'))
            TP += count_line_dict['TP']
            TN += count_line_dict['TN']
            FP += count_line_dict['FP']
            FN += count_line_dict['FN']

        scores['Macro Accuracy'] = (TP+TN) / (TP+TN+FP+FN)

        precision_correct = TP/(TP+FP)
        recall_correct = TP/(TP+FN)
        scores['F1_correct'] = 2*(precision_correct*recall_correct/(precision_correct+recall_correct))
        precision_wrong = TN/(TN+FN)
        recall_wrong = TN/(TN+FP)
        scores['F1_wrong'] = 2*(precision_wrong*recall_wrong/(precision_wrong+recall_wrong))

        scores = {k: [v] for k, v in scores.items()}
        scores = pd.DataFrame(scores)
        dump(scores, score_file)
        return scores
    
    @classmethod
    def evaluate_PRM(self, eval_file):
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', '_evaluate.xlsx')
        score_file = eval_file.replace(f'.{suffix}', '_score.csv')
        tmp_file = eval_file.replace(f'.{suffix}', '_evaluate.pkl')

        if not osp.exists(storage):
            raw_data = VL_PRMBenchmark('VL_PRMBenchmark').data
            data = load(eval_file)
            # data['prediction'] = [float(x) for x in data['prediction']]
            data['human_ranking'] = raw_data['response'].apply(lambda x: x['process_correctness'])

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

            data['count_line'] = [ans[idx] for idx in data['index']]
            # data.pop('image')
            dump(data, storage)

        data = load(storage)
        lt = len(data)

        category_scores = defaultdict(lambda: 0)
        category_cnt = defaultdict(lambda: 0)
        scores = defaultdict(lambda: 0)

        '''for i in range(lt):
            item = data.iloc[i]
            category_scores[item['category']] += item['score']
            category_cnt[item['category']] += 1
        # calculate the average score for each category
        for k, v in category_scores.items():
            scores[k] = v / category_cnt[k]
        # calculate category macro accuracy (average across categories)
        scores['Macro Accuracy'] = sum(scores.values()) / len(scores)
        # calculate the total average score
        scores['Overall Consistency'] = sum(category_scores.values()) / lt'''

        TP=0
        TN=0
        FP=0
        FN=0

        for i in range(lt):
            item = data.iloc[i]
            count_line_dict = json.loads(item['count_line'].replace("'", '"'))
            TP += count_line_dict['TP']
            TN += count_line_dict['TN']
            FP += count_line_dict['FP']
            FN += count_line_dict['FN']

        scores['Macro Accuracy'] = (TP+TN) / (TP+TN+FP+FN)

        precision_correct = TP/(TP+FP)
        recall_correct = TP/(TP+FN)
        scores['F1_correct'] = 2*(precision_correct*recall_correct/(precision_correct+recall_correct))
        precision_wrong = TN/(TN+FN)
        recall_wrong = TN/(TN+FP)
        scores['F1_wrong'] = 2*(precision_wrong*recall_wrong/(precision_wrong+recall_wrong))

        scores = {k: [v] for k, v in scores.items()}
        scores = pd.DataFrame(scores)
        dump(scores, score_file)
        return scores