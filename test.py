from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = 'Meta-Llama-3.1-8B-Instruct'
lora_path = './output/llama3_1_instruct_lora/checkpoint-18' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "从视觉文化或哲学角度分析欧洲难民问题"

messages = [
        {"role": "system", "content": "假设你是一位政治哲学评论家"},
        {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# print(input_ids)

model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('问题', prompt)
print('回答',response)