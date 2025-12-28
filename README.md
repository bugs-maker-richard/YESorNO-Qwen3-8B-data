# YESorNO-Qwen3-8B

## 模型介绍

本模型 **YESorNO-Qwen3-8B** 是基于 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` 进行微调的模型。

- **ModelScope ID**: `richardlin0625/YESorNO-Qwen3-8B`

---

## 1. 训练数据集处理部分

### 1.1 第一步
我们发现在数据集中的数字用‘***’符号做了模糊处理,于是我们用`<NUMBER>`标记替换所有“***”，这样模型能够把它视作通用的数字占位符。

### 1.2 第二步
为了能充分利用Qwen模型的思考能力，我们调用deepseekV3.2的API，让其根据label标签生成一段含有COT的推理过程，并规范其输出格式

这是我们用API生成COT的脚本：

```python
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
API_KEY = " API Key"
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# 输入和输出文件路径
INPUT_FILE = r"C:\Users\Linzhijian\Downloads\数据训练\train.json"
OUTPUT_FILE = r"C:\Users\Linzhijian\Downloads\数据训练\train_reason.json"

# 并发配置
MAX_WORKERS = 20 



client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


total_prompt_tokens = 0
total_completion_tokens = 0
token_lock = threading.Lock()

def generate_reasoning(item):
    """
    调用 DeepSeek 生成推理过程，返回处理后的完整 item 和 token 消耗
    """
 
    if 'text1' in item and 'text2' in item:
        text1 = item['text1']
        text2 = item['text2']
        label = item.get('label', '')
        
    
        text_input = f"句子1：{text1}\n句子2：{text2}"
      
        instruction = item.get('instruction', '请判断这两句话的语意是否一致，并说明理由。')
    else:
      
        text_input = item.get('input', '')
        label = item.get('output', '')
        instruction = item.get('instruction', '')

    if not text_input or not label:
        return None, 0, 0

    system_prompt = "你是一个花呗客服，请判断用户的这两句话语意是否一致。"
    
    user_prompt = f"""
任务：请分析给定两句话的语意是否相似，并解释判断理由。

输入数据：
{text_input}

【重要】已知这两句话的正确标签是："{label}"。

请你生成一段详细的思维链推理过程，解释为什么这两句话是"{label}"。

要求：
1. 先分析句子1的意图。
2. 再分析句子2的意图。
3. 对比两者的核心差异或共同点。
4. 最后输出结论。
5. 输出格式必须包含“推理过程：”和“结论：”。
6. 输出总字数控制在100字之内
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7, 
            max_tokens=1024
        )
        
        content = response.choices[0].message.content.strip()
        
       
        usage = response.usage
        p_tokens = usage.prompt_tokens
        c_tokens = usage.completion_tokens
        
        new_item = {
            "instruction": instruction,
            "input": text_input,
            "output": content
        }
        
        return new_item, p_tokens, c_tokens

    except Exception as e:
        print(f"\nAPI 调用出错 ({text_input[:10]}...): {e}")
        return None, 0, 0

def main():
    global total_prompt_tokens, total_completion_tokens

   
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
         
            data = json.load(f)
        except json.JSONDecodeError:
          
            f.seek(0)
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    print(f"成功加载 {len(data)} 条数据，开始并发生成推理过程 (并发数: {MAX_WORKERS})...")

    new_data = []
    
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    
        future_to_item = {executor.submit(generate_reasoning, item): item for item in data}
        
       
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Processing"):
            result_item, p_tokens, c_tokens = future.result()
            
            if result_item:
                new_data.append(result_item)
                
                with token_lock:
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens

   
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！")
    print(f"原始数据: {len(data)} 条")
    print(f"生成数据: {len(new_data)} 条")
    print(f"Token 消耗统计:")
    print(f"  - 输入 Tokens (Prompt): {total_prompt_tokens}")
    print(f"  - 输出 Tokens (Completion): {total_completion_tokens}")
    print(f"  - 总计 Tokens: {total_prompt_tokens + total_completion_tokens}")
    print(f"新文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
```

### 1.3 第三步
我们从训练集中抽取了1000条数据作为验证集，用于微调过程中的模型评估，剩余的数据作为训练集进行微调训练。

---

## 2. 关于我们是如何微调的

### 2.1 硬件
A100-80G-SXM*1，RAM 120G

### 2.2 运行环境
- **pytorch**: 2.8.0-cuda12.6-cudnn9-py311-ubuntu22.04
- **Transformers version**: 4.57.1
- **微调平台**：llamafactory 0.9.4
- **监控平台**：swanlab

### 2.3 微调参数(占用显存约60G）

#### 2.3.1 第一次微调
```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path PATH/OF/BASE/MODEL \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template Alpaca \
    --flash_attn fa2 \
    --dataset_dir data \
    --dataset train_alpaca_2w_reason \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 2.0 \
    --max_samples 20000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 30 \
    --warmup_steps 20 \
    --packing True \
    --enable_thinking False \
    --report_to none \
    --output_dir PATH/YOU/WANT/O/SAVE \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.03 \
    --lora_target all \
    --swanlab_api_key YOUR_KEY_HERE \
    --swanlab_mode cloud 
```

#### 2.3.2 第二次微调
（利用上一次的checkpoint以较小学习率对剩余1W条数据继续训练）
```bash
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path PATH/OF/BASE/MODEL \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template Alpaca \
    --flash_attn fa2 \
    --dataset_dir data \
    --dataset train_alpaca_1w_reason \
    --cutoff_len 1024 \
    --learning_rate 5e-01 \
    --num_train_epochs 1.0 \
    --max_samples 20000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 20 \
    --warmup_steps 10 \
    --packing True \
    --enable_thinking False \
    --report_to none \
    --output_dir PATH/YOU/WANT/O/SAVE \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.03 \
    --lora_target all \
    --swanlab_api_key YOUR_KEY_HERE \
    --swanlab_mode cloud 
```

### 2.4 合并lora权重到基础模型
并利用验证集评估效果

---

## 3. 推理部分

### 3.1 启动服务
我们用llamafactory启动API服务进行推理

### 3.2 推理思路
推理脚本大致思路如下：对测试集进行5轮推理，当5轮推理的结果中有3次及以上结果相同，则认为该结果为最终答案

### 3.3 推理脚本
```python
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "YESorNO-Qwen3-8B"
INPUT_FILE = "test.jsonl"
OUTPUT_DIR = "PATH/YOU/WANT/O/SAVE"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "inference_result.jsonl")


TOTAL_ROUNDS = 5 


VOTE_STRATEGY = 'strict'

SYSTEM_PROMPT = (
    "你是一位资深严谨的花呗金融客服，你的任务是判断客户的两句话在金融场景下语义是否相同。"
    "注意先严谨地重点分析并输出150字左右的这两句话的关键词、意图、主体和核心语义的异同。"
    "然后得出结论‘相同’或‘不相同’。\n"
)

def call_llm(sentence1, sentence2):
    """单次调用 API"""
    headers = {"Content-Type": "application/json"}
    user_content = f"句子1：{sentence1}\n句子2：{sentence2}"
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.6, 
        "top_p": 0.9,
        "stream": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
      
        if "不相同" in content: 
            return "不相同"
        elif "相同" in content: 
            return "相同"
            
        return "不相同" 
    except Exception as e:
        return "Error"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


    raw_data = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    total_items = len(raw_data)
    print(f"载入 {total_items} 条数据，准备进行 {TOTAL_ROUNDS} 轮推理...")

  
    all_votes_record = [[] for _ in range(total_items)]


    for round_num in range(1, TOTAL_ROUNDS + 1):
        print(f"\n======== 开始第 {round_num} / {TOTAL_ROUNDS} 轮推理 ========")
        

        for idx, item in enumerate(tqdm(raw_data, desc=f"Round {round_num}")):
            s1 = item.get("句子1", "")
            s2 = item.get("句子2", "")
            
            if s1 and s2:
     
                res = call_llm(s1, s2)
                
    
                if res != "Error":
                    all_votes_record[idx].append(res)
                
 
                print(f" [R{round_num}|ID:{idx+1}] {res} | {s1[:10]}... vs {s2[:10]}...")
        


    print(f"\n======== 所有 {TOTAL_ROUNDS} 轮推理结束，正在计算最终结果并写入 ========")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(raw_data):
            votes = all_votes_record[idx]
            
            if not votes:
                final_result = "Error"
            else:
                if VOTE_STRATEGY == 'strict':
                 
                    final_result = "不相同" if "不相同" in votes else "相同"
                else:
                  
                    final_result = Counter(votes).most_common(1)[0][0]
            
          
            print(f"ID:{idx+1} 投票详情:{votes} => 最终判定:【{final_result}】")

          
            result_item = {
                "句子1": item.get("句子1"),
                "句子2": item.get("句子2"),
                "response": final_result
            }
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    print(f"\n处理完成！最终文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
```
