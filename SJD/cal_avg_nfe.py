def parse_nfe_latency(file_path):
    total_nfe = 0
    total_latency = 0
    count = 0

    with open(file_path, 'r') as f:
        for line in f:
            if 'NFE:' in line and 'latency:' in line:
                try:
                    nfe_part = line.split('NFE:')[1].split(',')[0].strip()
                    latency_part = line.split('latency:')[1].strip()
                    if float(latency_part)<200:
                        nfe = int(nfe_part)
                        latency = float(latency_part)
                        total_nfe += nfe
                        total_latency += latency
                        count += 1
                except (IndexError, ValueError):
                    print(f"Failed to parse line: {line.strip()}")

    if count == 0:
        print("No valid samples found.")
        return

    avg_nfe = total_nfe / count
    avg_latency = total_latency / count

    print(f"Total samples: {count}")
    print(f"Average NFE: {avg_nfe:.2f}")
    print(f"Average latency: {avg_latency:.2f} ms")

# 使用时传入你的 txt 文件路径
file_path=input('input your nfe file: ')
parse_nfe_latency(file_path)
