import os
import copy
import re
import pandas as pd
from agent_workflow.graph import build_graph
from agent_workflow.config import MODEL_NAME

def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)
    return name.strip(' .')

def main():
    # 编译并获取 LangGraph 图
    graph = build_graph()
    
    # 确保输出目录存在
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{safe_filename(MODEL_NAME)}_output5.csv")
    
    # 读取 parquet 数据集
    dataset_path = "dataset/MCQ_sampled.parquet"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    # 保留原来的保存csv文件的逻辑：项目开始运行就直接创建并且每次运行写入结果
    if os.path.exists(output_file):
        res = pd.read_csv(output_file, index_col=0)
        last_idx = res['extracted_option'].last_valid_index()
        if last_idx is not None:
            last_valid_index = int(last_idx) + 1
        else:
            last_valid_index = 0
    else:
        res = copy.deepcopy(df)
        res['extracted_option'] = None
        res['output_text'] = None
        last_valid_index = 0
        # 第一次运行时就直接创建文件
        res.to_csv(output_file)
        
    # 遍历数据集
    for qa_idx in range(last_valid_index, res.shape[0]):
        row = res.iloc[qa_idx]
        question_id = row.get("Question_id")
        video_id = row.get("video_id")
        question = row.get("question")
        question_category = row.get("question_category", "")

        video_path = f"dataset/videos/{video_id}"

        # 初始化图状态
        initial_state = {
            "video_path": video_path,
            "question": question,
            "question_category": question_category,
            "messages": [],
            "retry_count": 0,
            "extracted_option": None,
            "spatial_memory": {}  # 初始化空间记忆字典
        }
        
        print(f"Processing index: {qa_idx}, Question_id: {question_id}, video: {video_id}")
        
        # 调用图
        try:
            final_state = graph.invoke(initial_state)
            
            extracted_option = final_state.get("extracted_option")
            messages = final_state.get("messages", [])
            output_text = ""
            if messages:
                # 获取最后一条消息的内容
                last_msg_content = messages[-1].content
                if isinstance(last_msg_content, list):
                    for item in last_msg_content:
                        if item.get("type") == "text":
                            output_text += item.get("text", "")
                else:
                    output_text = str(last_msg_content)
                    
            res.loc[res.index[qa_idx], 'extracted_option'] = extracted_option
            res.loc[res.index[qa_idx], 'output_text'] = output_text
            
            print(f"Result for {question_id}: extracted_option={extracted_option}")
            
        except Exception as e:
            print(f"Error processing Question_id {question_id}: {e}")
            res.loc[res.index[qa_idx], 'extracted_option'] = "ERROR"
            res.loc[res.index[qa_idx], 'output_text'] = str(e)
            
        # 每次运行写入结果
        res.to_csv(output_file)
        
    print(f"Processing completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
