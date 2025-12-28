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