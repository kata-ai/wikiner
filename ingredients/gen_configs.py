import sys
from itertools import product

from numpy.random import uniform
import yaml


def generate_params(config: dict):
    """
    pre-generate hyperparameter configs

    config: dict
        num_trials: int
        num_split: int
        rand_params: List[rand_instance]
            random_instance: dict 
                - name : str
                - min_value: float/int
                - max_value: float/int
        grid_params: List[grid_instance]
            grid_instance: dict
                - name: str
                - values: List
        e.g : from yaml
        num_trials: 2
        num_split: 2
        rand_params:
        - name: learning_rate
        min_value: 0.001
        max_value: 0.0041
        - name: dropout
        min_value: 0.01
        max_value: 0.76
        grid_params:
        - name: char_integration_method
        values:
        - attention
        - none
        - name: decay
        values:
        - True
        - False
    
    Return
    ------
    List[dict]
    """
    num_trials = config['num_trials']
    num_split = config['num_split']
    rand_params = config['rand_params'] if 'rand_params' in config else []
    grid_params = config['grid_params'] if 'grid_params' in config else []

    search_table = []
    for _ in range(num_split):
        gen_hyper = []
        colnames = []
        for i in rand_params+grid_params:
            colnames.append(i['name'])
        for i in range(num_trials//num_split):
            row = []
            for j in rand_params:
                low = j['min_value']
                high = j['max_value']
                if 'scale' in j:
                    row.append(j['scale']**uniform(low, high))
                else:
                    row.append(uniform(low, high))
            grids = []
            for k in grid_params:
                grids.append(k['values'])
            for k in product(*grids):
                for l in k:
                    row.append(l)
                gen_hyper.append(row.copy())
                row = row[:-len(k)]
        assert len(gen_hyper) == len(set([tuple(i) for i in gen_hyper]))
        for i in gen_hyper:
            param = {}
            for idx, col in enumerate(colnames):
                param[col] = i[idx]
            search_table.append(param)
    return search_table
        

if __name__ == '__main__':
    for cfg in generate_params(yaml.load(open(sys.argv[1]))):
        print(cfg)
    
