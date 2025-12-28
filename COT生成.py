import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "sk-9fd8c842b2e34f4a809a5fc9ee51eb9c"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-max"

# 输入和输出文件路径
INPUT_FILE = r"C:\Users\Linzhijian\Downloads\数据训练\2400right.json"
OUTPUT_FILE = r"C:\Users\Linzhijian\Downloads\数据训练\2400right_reason2.json"

# 并发配置
MAX_WORKERS = 20  # 同时进行的请求数量，根据API限制调整
# ===========================================

# 初始化客户端 (OpenAI 客户端是线程安全的)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 全局统计变量和锁
total_prompt_tokens = 0
total_completion_tokens = 0
token_lock = threading.Lock()

def generate_reasoning(item):
    """
    调用 DeepSeek 生成推理过程，返回处理后的完整 item 和 token 消耗
    """
    # 适配数据格式 {"text1": "...", "text2": "...", "label": "..."}
    if 'text1' in item and 'text2' in item:
        text1 = item['text1']
        text2 = item['text2']
        label = item.get('label', '')
        
        # 构造输入
        text_input = f"句子1：{text1}\n句子2：{text2}"
        # 如果原数据没有 instruction，给一个默认的
        instruction = item.get('instruction', '请判断这两句话的语意是否一致，并说明理由。')
    else:
        # 兼容旧格式 (input/output/instruction)
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
        
        # 获取 Token 消耗
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

    # 1. 读取原始数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            # 尝试作为整个 JSON 读取
            data = json.load(f)
        except json.JSONDecodeError:
            # 如果失败，尝试作为 JSON Lines 读取 (每行一个 JSON)
            f.seek(0)
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    print(f"成功加载 {len(data)} 条数据，开始并发生成推理过程 (并发数: {MAX_WORKERS})...")

    new_data = []
    
    # 2. 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(generate_reasoning, item): item for item in data}
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Processing"):
            result_item, p_tokens, c_tokens = future.result()
            
            if result_item:
                new_data.append(result_item)
                
                # 更新 Token 统计
                with token_lock:
                    total_prompt_tokens += p_tokens
                    total_completion_tokens += c_tokens

    # 4. 保存结果
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