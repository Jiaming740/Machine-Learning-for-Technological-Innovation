import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


# 限制文本长度，避免超出GPT-3.5的上下文长度
def get_aug_text_cn(client, text, max_length=6000):
    # 如果文本太长，进行截断
    if len(text) > max_length:
        text = text[:max_length]

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"请总结以下文本，突出主要的关键信息: {text}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return ''


def process_file(file_path):
    from openai import OpenAI
    client = OpenAI(
        base_url="https://hk.xty.app/v1",
        api_key="sk-rAwXE6SLBw2yuf4qCbD3B97c5c9f4b8fB34563C0Df549a01"
    )

    # 读取你提供的 TSV 文件
    csl_data = pd.read_csv(file_path, sep='\t')
    all_sampled_text = csl_data['abstract'].tolist()

    aug_data = []
    # 设置线程池的大小，例如10个线程
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交任务到线程池
        future_to_text = {executor.submit(get_aug_text_cn, client, text): text for text in all_sampled_text}
        # 进度条显示
        for future in tqdm(as_completed(future_to_text), total=len(future_to_text), desc=f"Processing {file_path}"):
            aug_text = future.result()
            aug_data.append(aug_text)

    # 将生成的aug_text直接覆盖原来的abstract列
    csl_data['abstract'] = aug_data

    # 设置输出路径
    output_dir = 'Training Data'  # 设置文件保存路径
    os.makedirs(output_dir, exist_ok=True)  # 如果路径不存在则创建

    # 输出为TSV文件
    output_file_name = os.path.basename(file_path).replace('.tsv', '_sum.tsv')
    output_file_path = os.path.join(output_dir, output_file_name)
    csl_data.to_csv(output_file_path, sep='\t', index=False)
    print(f"Completed processing {file_path}")
    return output_file_path


def main():
    file_path = 'Training Data/power_data_selfmedia.tsv'  # 输入文件路径
    augmented_file = process_file(file_path)
    print(f"summary data saved to {augmented_file}")


if __name__ == '__main__':
    main()

