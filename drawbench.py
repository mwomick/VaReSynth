import csv
from random import choices

def get_rand_drawbench_prompts(num, colors_only=False):
    with open('data/DrawBench.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        chs = choices(list(reader)[:24] if colors_only else list(reader), k=num)
        return [ prompt["Prompts"] for prompt in chs ]