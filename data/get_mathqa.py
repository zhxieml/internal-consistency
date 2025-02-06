# convert mathqa dataset to jsonl format, from https://math-qa.github.io/math-QA/ train.json
# manually choose 4 instances in the beginning as icl example and delete them in the end
import json
import ast
import re

def clean_options(option_string):
    # 正则表达式模式，用于匹配并移除 a )、b )、c )、d )、e ) 这部分
    pattern = re.compile(r'[a-e] \) ')
    
    try:
        # 尝试将字符串解析为列表
        option_list = ast.literal_eval(option_string)
        # 如果解析成功，移除每个选项前面的字母和括号，并去除多余的空格
        cleaned_list = [pattern.sub('', item).strip() for item in option_list]
    except (ValueError, SyntaxError):
        # 如果解析失败，说明它不是一个列表格式
        # 移除每个选项前面的字母和括号，并去除多余的空格
        cleaned_list = [pattern.sub('', item).strip() for item in option_string.split(",")]
    
    return cleaned_list

with open('train.json', 'r') as f:
    data = json.load(f)

new_data = []

for i, item in enumerate(data):
    if i >= 504:
        break

    new_item = {}
    
    new_item['question'] = item['Problem']

    new_item['answer'] = ord(item['correct'].lower()) - ord('a')

    new_item['choices'] = clean_options(item['options'])
        
    new_data.append(new_item)

with open('mathqa.jsonl', 'w') as f:
    for item in new_data:
        f.write(json.dumps(item) + '\n')