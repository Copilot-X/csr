import os
import base64
import glob
import csv
import time
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# ==========================================
# 1. 配置参数与路径设置
# ==========================================

MAX_WORKERS = 5      # 设置并发线程数
MAX_RETRIES = 10      # 接口请求失败时的最大重试次数
RETRY_DELAY = 15      # 重试前的等待时间（秒）

# 目录配置
INPUT_DIR = "data/pic"                     # 赛方规定的图片输入目录
CACHE_DIR = "cache_results"                # 缓存目录，用于断点续传
OUTPUT_DIR = "submission"                  # 最终提交包的目录
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "submission.csv")

# ==========================================
# 2. 辅助函数：图片转 Base64
# ==========================================
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# 3. 单张图片处理函数 (调用 DP Tech 接口 + 重试机制)
# ==========================================
def process_single_image(image_path, cache_path):
    base64_image = encode_image_to_base64(image_path)
    file_name = os.path.basename(image_path)
    print(f"🚀 线程启动: 正在处理 {file_name} ...")
    
    url = "https://ocsr.dp.tech/mol/img2mol"
    headers = {
        "accept": "*/*",
        "content-type": "application/json",
        "referrer": "https://ocsr.dp.tech/"
    }
    payload = {
        "base64_img": base64_image
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            # 发起 POST 请求，增加 timeout 防止长时间挂起
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 500:
                print(f"🚨 拦截到 500 Internal Server Error ({file_name})，服务端处理异常，已跳出重试！")
                return False

            response.raise_for_status()  # 检查 HTTP 状态码是否为 200
            
            result = response.json()
            
            # 判断接口返回的业务状态码 code 是否为 0 (0通常代表成功)
            if result.get("code") == 0:
                e_smiles = result.get("data", [])[0]['caption']
                score = result.get("data", [])[0]['score']
                
                # 写入 txt 缓存文件
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(e_smiles + "\t" + str(score))
                    
                print(f"✅ 成功: {file_name} -> 缓存完成")
                return True
            else:
                msg = result.get("msg", "未知错误")
                print(f"⚠️ 接口返回业务错误 ({file_name}): {msg}")
                # 遇到业务报错同样进入下方的重试逻辑（防并发误杀）
                
        except Exception as e:
            print(f"❌ 请求异常 ({file_name}) - 第 {attempt + 1} 次尝试失败: {e}")
            
        # 如果还没到最大重试次数，则等待一段时间后重试
        if attempt < MAX_RETRIES - 1:
            print(f"⏳ 等待 {RETRY_DELAY} 秒后对 {file_name} 进行第 {attempt + 2} 次重试...")
            time.sleep(RETRY_DELAY)
            
    print(f"🚨 彻底失败: {file_name} 已超过最大重试次数 ({MAX_RETRIES}次)。")
    return False

# ==========================================
# 4. 汇总生成 CSV 及 Zip 打包
# ==========================================
def generate_submission(image_files):
    print(f"\n" + "="*50)
    print(f"📄 开始生成最终的 submission 文件及压缩包 ...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 写入 CSV
    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'e_smiles']) # 写入表头
        
        for img_path in image_files:
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(file_name)[0]
            cache_path = os.path.join(CACHE_DIR, f"{base_name}.txt")
            
            e_smiles = ""
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as cf:
                    e_smiles = cf.read().strip()
            
            writer.writerow([file_name, e_smiles])
            
    print(f"✅ CSV 结果已保存至: {OUTPUT_CSV}")

    # # 2. 生成预置的 meta.md（防止忘记写导致不符合赛方格式）
    # meta_path = os.path.join(OUTPUT_DIR, "meta.md")
    # if not os.path.exists(meta_path):
    #     with open(meta_path, 'w', encoding='utf-8') as f:
    #         f.write("# Meta Data\n\n请在此处补充比赛要求的 meta 信息（如果不需要可直接删除此段）。")
    #     print(f"✅ 已自动生成占位符文件: {meta_path}")

    # 3. 将整个 OUTPUT_DIR 打包为 zip
    zip_filename = "submission"  # shutil 会自动追加 .zip 后缀
    shutil.make_archive(zip_filename, 'zip', OUTPUT_DIR)
    print(f"📦 已将 {OUTPUT_DIR} 目录完整打包为: {zip_filename}.zip")

# ==========================================
# 5. 主执行循环 (多线程并发控制)
# ==========================================
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"错误：输入文件夹 '{INPUT_DIR}' 不存在。请确保图片放在对应的目录。")
        return

    # 获取所有图片文件
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper()))) 
    
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"在 '{INPUT_DIR}' 中没有找到图片文件。")
        return
        
    print(f"找到 {len(image_files)} 张图片，开始多线程批量处理 (最大线程数: {MAX_WORKERS})...\n" + "-"*50)
    
    success_count = 0
    failed_files = [] # 追踪所有失败的文件名
    
    # 阶段一：任务预过滤（断点续传拦截）
    tasks_to_run = []
    for idx, img_path in enumerate(image_files):
        # if idx < 652:
        #     continue
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cache_path = os.path.join(CACHE_DIR, f"{base_name}.txt")
        
        if os.path.exists(cache_path):
            print(f"⏩ 跳过: {base_name}.txt 缓存已存在。")
            success_count += 1
        else:
            tasks_to_run.append((img_path, cache_path))
            
    print(f"\n待处理的新任务数: {len(tasks_to_run)}")
    print("-" * 50)
    
    # 阶段二：提交给线程池
    if tasks_to_run:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_img = {
                executor.submit(process_single_image, img_path, cache_path): img_path 
                for img_path, cache_path in tasks_to_run
            }
            
            for future in as_completed(future_to_img):
                img_path = future_to_img[future]
                file_name = os.path.basename(img_path)
                try:
                    is_success = future.result()
                    if is_success:
                        success_count += 1
                    else:
                        failed_files.append(file_name) # 函数内用尽了重试次数，依然失败
                except Exception as exc:
                    print(f"❌ 线程池执行遇到严重异常 ({file_name}): {exc}")
                    failed_files.append(file_name) # 捕获意外的崩溃

    # 阶段三：合并结果并打包
    generate_submission(image_files)

    # 阶段四：统计与总结
    print("-" * 50)
    print(f"🎉 处理完成！共成功处理并合成了 {success_count} / {len(image_files)} 个结果。")
    
    # 输出异常统计列表
    if failed_files:
        print(f"\n⚠️ 警告：有 {len(failed_files)} 张图片未能成功识别！清单如下：")
        for bad_file in failed_files:
            print(f"  - {bad_file}")
    else:
        print(f"\n🌟 完美！所有新提交的图片都已100%成功识别。")
    
    print(f"\n💡 最终检查提醒：")
    print(f"1. 如需补充或修改 meta.md 内容，请进入 {OUTPUT_DIR} 修改后，手动重新压缩。")
    print(f"2. 如果不需要修改 meta 信息，直接提交当前目录下的 submission.zip 即可！")

if __name__ == "__main__":
    main()