#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT字幕文件转纯文本工具
支持批量处理和多种输出格式
"""

import argparse
import os
import re
from pathlib import Path


class SRTConverter:
    def __init__(self):
        self.time_pattern = re.compile(
            r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        )
        self.bracket_pattern = re.compile(r"\[.*?\]")
        self.parenthesis_pattern = re.compile(r"\(.*?\)")

    def clean_text(self, text, remove_brackets=True, remove_parenthesis=False):
        """清理文本，去除不需要的标记"""
        if remove_brackets:
            text = self.bracket_pattern.sub("", text)
        if remove_parenthesis:
            text = self.parenthesis_pattern.sub("", text)
        return text.strip()

    def parse_srt(
        self, srt_content, remove_brackets=True, remove_parenthesis=False, **kwargs
    ):
        """解析SRT内容，提取纯文本"""
        lines = srt_content.strip().split("\n")
        text_lines = []

        for line in lines:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 跳过序号行（纯数字）
            if line.isdigit():
                continue

            # 跳过时间码行
            if self.time_pattern.search(line):
                continue

            # 处理字幕文本
            clean_line = self.clean_text(line, remove_brackets, remove_parenthesis)
            if clean_line:
                text_lines.append(clean_line)

        return text_lines

    def convert_to_text(self, srt_content, format_type="continuous", **kwargs):
        """
        转换SRT为不同格式的文本

        format_type 选项:
        - 'continuous': 连续文本
        - 'paragraph': 分段文本
        - 'sentence': 每句一行
        - 'timestamp': 保留时间信息
        """
        # 从kwargs中分离出parse_srt需要的参数
        parse_kwargs = {
            "remove_brackets": kwargs.get("remove_brackets", True),
            "remove_parenthesis": kwargs.get("remove_parenthesis", False),
        }

        text_lines = self.parse_srt(srt_content, **parse_kwargs)

        if format_type == "continuous":
            return " ".join(text_lines)

        elif format_type == "paragraph":
            sentences_per_paragraph = kwargs.get("sentences_per_paragraph", 3)
            text = " ".join(text_lines)
            sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

            paragraphs = []
            for i in range(0, len(sentences), sentences_per_paragraph):
                paragraph = ". ".join(sentences[i : i + sentences_per_paragraph])
                if paragraph:
                    paragraphs.append(paragraph + ".")

            return "\n\n".join(paragraphs)

        elif format_type == "sentence":
            return "\n".join(text_lines)

        elif format_type == "timestamp":
            return self._convert_with_timestamp(srt_content, **parse_kwargs)

        else:
            raise ValueError(f"不支持的格式类型: {format_type}")

    def _convert_with_timestamp(
        self, srt_content, remove_brackets=True, remove_parenthesis=False
    ):
        """保留时间戳的转换"""
        lines = srt_content.strip().split("\n")
        result = []
        current_time = ""

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if self.time_pattern.search(line):
                current_time = line.split(" --> ")[0]
                continue

            if not line.isdigit():
                clean_line = self.clean_text(line, remove_brackets, remove_parenthesis)
                if clean_line:
                    result.append(f"[{current_time}] {clean_line}")

        return "\n".join(result)

    def convert_file(self, input_path, output_path=None, **kwargs):
        """转换文件"""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")

        # 读取SRT文件
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                srt_content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(input_path, "r", encoding="gbk") as f:
                srt_content = f.read()

        # 转换文本
        converted_text = self.convert_to_text(srt_content, **kwargs)

        # 确定输出路径
        if output_path is None:
            output_path = input_path.with_suffix(".txt")
        else:
            output_path = Path(output_path)

        # 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(converted_text)

        return output_path

    def batch_convert(self, input_dir, output_dir=None, **kwargs):
        """批量转换目录中的所有SRT文件"""
        input_dir = Path(input_dir)

        if not input_dir.is_dir():
            raise NotADirectoryError(f"目录不存在: {input_dir}")

        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        srt_files = list(input_dir.glob("*.srt"))
        if not srt_files:
            print(f"在 {input_dir} 中没有找到SRT文件")
            return []

        converted_files = []
        for srt_file in srt_files:
            try:
                output_file = output_dir / f"{srt_file.stem}.txt"
                self.convert_file(srt_file, output_file, **kwargs)
                converted_files.append(output_file)
                print(f"已转换: {srt_file.name} -> {output_file.name}")
            except Exception as e:
                print(f"转换失败 {srt_file.name}: {e}")

        return converted_files


def main():
    parser = argparse.ArgumentParser(description="SRT字幕文件转纯文本工具")
    parser.add_argument("input", help="输入SRT文件或目录路径")
    parser.add_argument("-o", "--output", help="输出文件或目录路径")
    parser.add_argument(
        "-f",
        "--format",
        choices=["continuous", "paragraph", "sentence", "timestamp"],
        default="continuous",
        help="输出格式",
    )
    parser.add_argument(
        "--sentences-per-paragraph", type=int, default=3, help="分段模式下每段的句子数"
    )
    parser.add_argument(
        "--remove-brackets", action="store_true", default=True, help="移除方括号内容"
    )
    parser.add_argument(
        "--remove-parenthesis",
        action="store_true",
        default=False,
        help="移除圆括号内容",
    )
    parser.add_argument(
        "--batch", action="store_true", help="批量处理目录中的所有SRT文件"
    )

    args = parser.parse_args()

    converter = SRTConverter()

    try:
        if args.batch or Path(args.input).is_dir():
            converted_files = converter.batch_convert(
                args.input,
                args.output,
                format_type=args.format,
                sentences_per_paragraph=args.sentences_per_paragraph,
                remove_brackets=args.remove_brackets,
                remove_parenthesis=args.remove_parenthesis,
            )
            print(f"\n成功转换 {len(converted_files)} 个文件")
        else:
            output_file = converter.convert_file(
                args.input,
                args.output,
                format_type=args.format,
                sentences_per_paragraph=args.sentences_per_paragraph,
                remove_brackets=args.remove_brackets,
                remove_parenthesis=args.remove_parenthesis,
            )
            print(f"转换完成: {output_file}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


# 使用示例
if __name__ == "__main__":
    # 如果作为脚本运行，执行命令行界面
    if len(os.sys.argv) > 1:
        exit(main())

    # 演示用法
    converter = SRTConverter()

    # 示例SRT内容
    sample_srt = """1
00:00:00,000 --> 00:00:05,000
 [no speech detected]

2
00:00:05,000 --> 00:00:10,000
 [no speech detected]

3
00:00:25,000 --> 00:00:32,000
 Okay, today we go more in details into the image

4
00:00:32,000 --> 00:00:35,000
 problem which is like our key problem we want to solve

5
00:00:35,000 --> 00:00:39,000
 to better understand the use of these models.

6
00:00:39,000 --> 00:00:44,000
 This is one of the simplest way the noise is,

7
00:00:44,000 --> 00:00:47,000
 it's one of the simplest way you have to assess whether,

8
00:00:47,000 --> 00:00:52,000
 to quantitatively assess whether an image prior is effective or not,

9
00:00:52,000 --> 00:00:58,000
 because it's one of the simplest problem where you need a prior."""

    print("=== 演示不同转换格式 ===\n")

    # 连续文本
    print("1. 连续文本:")
    print(converter.convert_to_text(sample_srt, "continuous"))
    print()

    # 分段文本
    print("2. 分段文本:")
    print(converter.convert_to_text(sample_srt, "paragraph", sentences_per_paragraph=2))
    print()

    # 每句一行
    print("3. 每句一行:")
    print(converter.convert_to_text(sample_srt, "sentence"))
    print()

    # 带时间戳
    print("4. 带时间戳:")
    print(converter.convert_to_text(sample_srt, "timestamp"))
    print()

    print("=== 使用说明 ===")
    print("命令行用法:")
    print("python srt_converter.py input.srt -o output.txt -f paragraph")
    print("python srt_converter.py ./srt_folder --batch -f continuous")
