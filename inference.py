import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

# # 环境变量设置
os.environ['IMAGE_MAX_TOKEN_NUM'] = '2048'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# 引入 TransformersEngine
from swift import RequestConfig, InferRequest, TransformersEngine, BaseArguments
from swift.template.vision_utils import load_image

def parse_args():
    parser = argparse.ArgumentParser(description=" Molecule E-SMILES Batch Inference using PT (TransformersEngine)")
    parser.add_argument("--model_path", type=str, default='pretrained_models/Qwen3.6-27B', help="Base model path")
    parser.add_argument("--adapter_path", type=str, default='output/qwen3.6-27b/checkpoint-15000', help="Path to the adapter checkpoint")
    # 修改默认测试集路径为你指定的路径
    parser.add_argument("--target_dir", type=str, default="data/pic", help="Directory of images")
    # 输出文件改为 csv
    parser.add_argument("--output_file", type=str, default="submission/submission.csv", help="Path to save result csv")
    # 转换为 PT 推理后，建议 max_batch_size 设为 1 保持稳定
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for TransformersEngine")
    return parser.parse_args()

def run_molecule_inference(args):
    # 1. 初始化 TransformersEngine 引擎
    print(f"🚀 Initializing TransformersEngine...")
    model_id_or_path = args.model_path

    # 参考 PT 样例配置引擎
    engine = TransformersEngine(
        model_id_or_path,
        adapters=[args.adapter_path],
        max_batch_size=args.batch_size
    )

    # 分子式通常较长，建议 max_tokens 设大一点（比如 4096），temperature 保持 0 追求确定性
    # 额外参考样例加入 num_beams 参数（若有需要可自行调整，这里保留你原本的低温度确定性策略）
    # request_config = RequestConfig(max_tokens=512, temperature=0)
    request_config = RequestConfig()

    # 2. 扫描图片
    all_tasks = []
    for root, _, files in os.walk(args.target_dir):
        files = sorted(files)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 注意：这里直接保留完整文件名 file，不再去后缀
                full_path = os.path.join(root, file)
                all_tasks.append((file, full_path))

    total_count = len(all_tasks)
    print(f"🔍 Found {total_count} images in {args.target_dir}. Starting inference...")

    results = []

    # 3. 批量推理（保持原有逻辑骨架，兼容 args.batch_size 块处理）
    for i in tqdm(range(0, total_count, args.batch_size), desc="Processing Batches"):
        if i > 10:
            break
        batch = all_tasks[i : i + args.batch_size]
        
        infer_reqs = []
        valid_files = []

        for file_name, img_path in batch:
            try:
                # Prompt 必须与 SFT 构建阶段严格一致
                messages = [{"role": "user", "content": """<image>请识别图中的分子，并以Extended-SMILES的形式输出预测结果。
                        
### Extended-SMILES (E-SMILES) 规范要求（必须严格遵守）：
基本格式必须为：`SMILES<sep>EXTENSION`
1. **SMILES**：前半部分是与 RDKit 兼容的标准 SMILES 字符串。
2. **<sep>**：作为特殊分隔符，将常规 SMILES 与扩展描述分开（若无扩展结构，视情况保留或省略均可，但建议统一格式）。
3. **EXTENSION**：后半部分使用类似 XML 的特殊标记来描述复杂结构（如马库什结构等）：
   - 取代基/缩写基团: 格式为 `<a> [ATOM_INDEX]: [GROUP_NAME] </a>`。例如：`<a>0:R[1]</a>` 或 `<a>12:Ph</a>`。
   - 位置不确定的环取代物: 格式为 `<r> [RING_INDEX]: [GROUP_NAME] </r>`。
   - 抽象环: 格式为 `<c> [CIRCLE_INDEX]: [CIRCLE_NAME] </c>`。
   - 连接点: 使用特殊标记 `<dum>`，如 `<a>0:<dum></a>`。   

最终输出该分子的 Extended-SMILES 字符串。                
"""}]
                infer_reqs.append(InferRequest(messages=messages, images=[img_path]))
                valid_files.append(file_name)
            except Exception as e:
                print(f"❌ Error loading {img_path}: {e}")
                # 赛题规定：报错或放弃的留空
                results.append({"file_name": file_name, "e_smiles": ""})

        if infer_reqs:
            try:
                # 使用 TransformersEngine 进行推理
                resp_list = engine.infer(infer_reqs, request_config)
                
                for file_name, resp in zip(valid_files, resp_list):
                    content = resp.choices[0].message.content
                    
                    # 兼容可能带有 reasoning 思维链的模型
                    if "</think>" in content:
                        content = content.split("</think>")[-1]
                    else:
                        content = content
                        
                    results.append({"file_name": file_name, "e_smiles": content})
            except Exception as e:
                print(f"⚠️ Batch inference error: {e}")
                for file_name in valid_files:
                    # 批次推理崩溃时，该批次全部填空字符串，保证行数不丢失
                    results.append({"file_name": file_name, "e_smiles": ""})

    # 4. 保存为 CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # 使用 Pandas 导出标准 CSV，表头为 file_name, e_smiles
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False, encoding='utf-8')

    print(f"✅ Finished! Saved {len(results)} results to {args.output_file}")
    print("💡 温馨提示：提交前请记得将生成的 csv 与 meta.md 一同打包为 zip。")

if __name__ == "__main__":
    args = parse_args()
    run_molecule_inference(args)