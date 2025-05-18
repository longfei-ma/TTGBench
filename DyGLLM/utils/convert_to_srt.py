import sys
from datetime import timedelta

def parse_timestamp(timestamp):
    """
    解析时间戳，返回一个 timedelta 对象。
    支持两种格式：
    - 分钟:秒（如 0:03）
    - 小时:分钟:秒（如 1:02:03）
    """
    parts = timestamp.split(':')
    if len(parts) == 2:  # 格式为 分钟:秒
        minutes, seconds = map(int, parts)
        return timedelta(minutes=minutes, seconds=seconds)
    elif len(parts) == 3:  # 格式为 小时:分钟:秒
        hours, minutes, seconds = map(int, parts)
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

def format_timestamp(delta):
    """
    将 timedelta 对象格式化为 SRT 时间轴格式（HH:MM:SS,000）。
    """
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},000"

def convert_to_srt(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        subtitle_number = 1

        # 用于存储最近的两行字幕
        previous_subtitle = None

        for i in range(0, len(lines), 2):  # 每两行处理一个字幕
            if i + 1 >= len(lines):
                break  # 如果文件行数不完整，跳过最后一行

            # 提取时间戳和字幕内容
            timestamp_line = lines[i].strip()
            content_line = lines[i + 1].strip()

            # 解析当前字幕的开始时间
            start_time = parse_timestamp(timestamp_line)

            # 解析下一字幕的开始时间作为当前字幕的结束时间
            if i + 2 < len(lines):
                next_timestamp_line = lines[i + 2].strip()
                end_time = parse_timestamp(next_timestamp_line)
            else:
                # 如果是最后一行字幕，结束时间设置为开始时间加20秒
                end_time = start_time + timedelta(seconds=20)

            # 格式化时间轴
            start_time_str = format_timestamp(start_time)
            end_time_str = format_timestamp(end_time)

            # 生成当前字幕的内容
            if previous_subtitle:
                # 如果存在上一行字幕，则显示上一行和当前行
                current_subtitle = f"{previous_subtitle}\n{content_line}"
            else:
                # 如果是第一行字幕，只显示当前行
                current_subtitle = content_line

            # 写入 SRT 文件
            outfile.write(f"{subtitle_number}\n")
            outfile.write(f"{start_time_str} --> {end_time_str}\n")
            outfile.write(f"{current_subtitle}\n\n")

            # 更新最近的两行字幕
            previous_subtitle = content_line
            subtitle_number += 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_srt.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_to_srt(input_file, output_file)
    print(f"SRT file saved to {output_file}")