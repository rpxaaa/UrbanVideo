"""Parallel batch runner for UrbanVideo-Bench.

Processes ALL questions concurrently using a global thread pool (default 4 workers).
Spatial memory is built programmatically inside the Reasoner node (no LLM-based
pre-computation needed). Supports resume from saved progress.

Usage:
    python batch_run.py                          # sampled dataset, 4 workers
    python batch_run.py --full --workers 8       # full dataset, 8 workers
"""

import os
import re
import copy
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_workflow.graph import build_graph
from agent_workflow.config import MODEL_NAME


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)
    return name.strip(' .')


def process_one_question(video_path: str, question: str, question_category: str) -> dict:
    """Run the graph for a single question. Returns result dict."""
    initial_state = {
        "video_path": video_path,
        "question": question,
        "question_category": question_category,
        "messages": [],
        "retry_count": 0,
        "extracted_option": None,
        "spatial_memory": {},
        "evidence_summary": None,
        "verification_passed": False,
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    extracted_option = final_state.get("extracted_option")
    messages = final_state.get("messages", [])
    output_text = ""
    if messages:
        last_msg_content = messages[-1].content
        if isinstance(last_msg_content, list):
            for item in last_msg_content:
                if item.get("type") == "text":
                    output_text += item.get("text", "")
        else:
            output_text = str(last_msg_content)

    return {
        "extracted_option": extracted_option,
        "output_text": output_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Use full MCQ dataset")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{safe_filename(MODEL_NAME)}_output_5_27_1.csv")

    dataset_path = "dataset/MCQ.parquet" if args.full else "dataset/MCQ_500_normal(1).parquet"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"Total questions: {len(df)}")

    # Initialize or resume
    if os.path.exists(output_file):
        res = pd.read_csv(output_file, index_col=0)
        # Find last processed row: extracted_option is non-empty and non-NaN
        opt = res['extracted_option']
        processed = opt.notna() & (opt.astype(str).str.strip() != '') & (opt.astype(str).str.strip() != 'nan')
        if processed.any():
            start_idx = int(processed[::-1].idxmax()) + 1
        else:
            start_idx = 0
    else:
        res = copy.deepcopy(df)
        res['extracted_option'] = pd.array([None] * len(res), dtype='object')
        res['output_text'] = pd.array([None] * len(res), dtype='object')
        res.to_csv(output_file)
        start_idx = 0

    if start_idx >= len(res):
        print("All questions already processed.")
        return

    remaining = res.iloc[start_idx:]
    remaining_indices = remaining.index.tolist()

    print(f"Remaining: {len(remaining)} questions across {remaining['video_id'].nunique()} videos")
    print(f"Workers: {args.workers}\n")

    completed_count = 0
    save_interval = max(len(remaining_indices) // 20, 5)  # Save every ~5% progress

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx in remaining_indices:
            row = res.iloc[idx]
            video_path = f"dataset/videos/{row['video_id']}"
            future = executor.submit(
                process_one_question,
                video_path, row['question'], row['question_category']
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            row = res.iloc[idx]
            try:
                result = future.result()
                res.loc[idx, 'extracted_option'] = result['extracted_option']
                res.loc[idx, 'output_text'] = result['output_text']
                print(f"[{idx}] Q={row['Question_id']} cat={row['question_category']} -> {result['extracted_option']}")
            except Exception as e:
                print(f"[{idx}] Q={row['Question_id']} Error: {e}")
                res.loc[idx, 'extracted_option'] = "ERROR"
                res.loc[idx, 'output_text'] = str(e)

            completed_count += 1
            if completed_count % save_interval == 0:
                res.to_csv(output_file)

    # Final save
    res.to_csv(output_file)
    print(f"\nDone. Results saved to {output_file}")


if __name__ == "__main__":
    main()
