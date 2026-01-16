import math

def split_text_file(input_file, num_splits=4):
    # 1. 读取所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    total_lines = len(lines)
    # 计算每份的基础大小
    avg = total_lines / num_splits
    
    print(f"总行数: {total_lines}, 预计每份约: {avg:.2f} 行")

    # 2. 切分并写入文件
    for i in range(num_splits):
        # 计算当前块的起始和结束索引
        start = int(math.ceil(i * avg))
        end = int(math.ceil((i + 1) * avg))
        chunk = lines[start:end]
        
        # 生成文件名，例如: split_1.txt, split_2.txt ...
        output_filename = f"process_list{i+1}.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            out_f.write('\n'.join(chunk))
        
        print(f"已生成 {output_filename} ({len(chunk)} 行)")

if __name__ == "__main__":
    # 请确保你的文件名正确
    split_text_file('process_list_wrongcase.txt', 4)