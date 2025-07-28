#!/usr/bin/env python3
"""
新颖性检查脚本 - 检查生成文本是否来自训练数据的记忆

功能：
1. N-gram重复检查 - 检查长序列是否在训练数据中出现
2. 完整文本匹配 - 检查生成的文本是否完全匹配训练数据
3. 模糊匹配 - 使用编辑距离检查相似文本
4. 统计分析 - 分析重复程度和新颖性指标
"""

import argparse
import os
import sys
import json
import re
from collections import defaultdict, Counter
from typing import List, Set, Tuple, Dict
import numpy as np
from difflib import SequenceMatcher

# 添加项目根目录到路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cs336_basics.tokenizer import Tokenizer


def load_training_data(data_path: str, tokenizer: Tokenizer, max_samples: int = None) -> List[str]:
    """
    加载训练数据并解码为文本
    
    Args:
        data_path (str): 训练数据路径（.npy格式的token文件）
        tokenizer (Tokenizer): tokenizer对象
        max_samples (int, optional): 最大加载样本数，用于大数据集
        
    Returns:
        List[str]: 解码后的文本列表
    """
    print(f"Loading training data from: {data_path}")
    
    if data_path.endswith('.npy'):
        # 加载tokenized数据
        tokens = np.load(data_path)
        print(f"Loaded {len(tokens)} tokens")
        
        # 将tokens分割为文档（使用endoftext token）
        endoftext_bytes = "<|endoftext|>".encode("utf-8")
        endoftext_id = tokenizer.token_to_id.get(endoftext_bytes, None)
        
        # 检查是否实际存在endoftext token
        endoftext_count = np.sum(tokens == endoftext_id) if endoftext_id is not None else 0
        
        if endoftext_id is None or endoftext_count == 0:
            # 如果没有endoftext token或token不存在，按固定长度分割
            chunk_size = 512  # 每个文档512个token
            documents = []
            print(f"No endoftext tokens found, splitting into {chunk_size}-token chunks...")
            
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i+chunk_size]
                if len(chunk) > 50:  # 至少50个token
                    try:
                        text = tokenizer.decode(chunk.tolist())
                        if len(text.strip()) > 100:  # 至少100个字符
                            documents.append(text.strip())
                    except Exception as e:
                        print(f"Decode error at chunk {i//chunk_size}: {e}")
                        continue
                
                if max_samples and len(documents) >= max_samples:
                    print(f"Reached max_samples limit: {max_samples}")
                    break
                    
                # 每1000个文档打印一次进度
                if len(documents) % 1000 == 0 and len(documents) > 0:
                    print(f"Processed {len(documents)} documents...")
        else:
            # 使用endoftext token分割
            documents = []
            start = 0
            print(f"Looking for endoftext token {endoftext_id} in {len(tokens)} tokens...")
            
            for i, token in enumerate(tokens):
                if token == endoftext_id:
                    if i > start:
                        chunk = tokens[start:i]
                        if len(chunk) > 10:  # 至少10个token
                            try:
                                text = tokenizer.decode(chunk.tolist())
                                if len(text.strip()) > 20:  # 至少20个字符
                                    documents.append(text.strip())
                            except Exception as e:
                                print(f"Decode error at position {i}: {e}")
                                continue
                    start = i + 1
                    if max_samples and len(documents) >= max_samples:
                        print(f"Reached max_samples limit: {max_samples}")
                        break
                    
                    # 每1000个文档打印一次进度
                    if len(documents) % 1000 == 0 and len(documents) > 0:
                        print(f"Processed {len(documents)} documents...")
            
            # 处理最后一个文档（如果没有以endoftext结尾）
            if start < len(tokens) and len(documents) < (max_samples or float('inf')):
                chunk = tokens[start:]
                if len(chunk) > 10:
                    try:
                        text = tokenizer.decode(chunk.tolist())
                        if len(text.strip()) > 20:
                            documents.append(text.strip())
                    except Exception as e:
                        print(f"Decode error for final chunk: {e}")
    
    elif data_path.endswith('.txt'):
        # 加载原始文本文件
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按段落或固定长度分割
        documents = content.split('\n\n')  # 按双换行分割
        documents = [doc.strip() for doc in documents if len(doc.strip()) > 10]
        
        if max_samples:
            documents = documents[:max_samples]
    
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Processed {len(documents)} documents")
    return documents


def extract_ngrams(text: str, n: int) -> Set[str]:
    """
    提取文本的n-gram
    
    Args:
        text (str): 输入文本
        n (int): n-gram的长度
        
    Returns:
        Set[str]: n-gram集合
    """
    # 简单的单词级n-gram
    words = text.lower().split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    return ngrams


def extract_char_ngrams(text: str, n: int) -> Set[str]:
    """
    提取字符级n-gram
    
    Args:
        text (str): 输入文本
        n (int): n-gram的长度
        
    Returns:
        Set[str]: 字符级n-gram集合
    """
    text = text.lower()
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams.add(ngram)
    return ngrams


def check_exact_match(generated_text: str, training_docs: List[str]) -> Tuple[bool, List[str]]:
    """
    检查完整文本匹配
    
    Args:
        generated_text (str): 生成的文本
        training_docs (List[str]): 训练文档列表
        
    Returns:
        Tuple[bool, List[str]]: (是否匹配, 匹配的文档列表)
    """
    generated_clean = generated_text.strip().lower()
    matches = []
    
    for doc in training_docs:
        doc_clean = doc.strip().lower()
        if generated_clean == doc_clean:
            matches.append(doc)
        elif generated_clean in doc_clean:
            matches.append(doc)
    
    return len(matches) > 0, matches


def check_ngram_overlap(generated_text: str, training_docs: List[str], 
                       n_values: List[int] = [4, 6, 8, 10]) -> Dict[int, float]:
    """
    检查n-gram重叠
    
    Args:
        generated_text (str): 生成的文本
        training_docs (List[str]): 训练文档列表
        n_values (List[int]): 要检查的n-gram长度列表
        
    Returns:
        Dict[int, float]: 每个n值的重叠比例
    """
    results = {}
    
    for n in n_values:
        generated_ngrams = extract_ngrams(generated_text, n)
        if not generated_ngrams:
            results[n] = 0.0
            continue
        
        # 构建训练数据的n-gram集合
        training_ngrams = set()
        for doc in training_docs:
            training_ngrams.update(extract_ngrams(doc, n))
        
        # 计算重叠
        overlap = generated_ngrams.intersection(training_ngrams)
        overlap_ratio = len(overlap) / len(generated_ngrams)
        results[n] = overlap_ratio
    
    return results


def check_char_ngram_overlap(generated_text: str, training_docs: List[str],
                            n_values: List[int] = [15, 20, 25, 30]) -> Dict[int, float]:
    """
    检查字符级n-gram重叠
    
    Args:
        generated_text (str): 生成的文本
        training_docs (List[str]): 训练文档列表
        n_values (List[int]): 要检查的字符n-gram长度列表
        
    Returns:
        Dict[int, float]: 每个n值的重叠比例
    """
    results = {}
    
    for n in n_values:
        generated_ngrams = extract_char_ngrams(generated_text, n)
        if not generated_ngrams:
            results[n] = 0.0
            continue
        
        # 构建训练数据的字符n-gram集合
        training_ngrams = set()
        for doc in training_docs:
            training_ngrams.update(extract_char_ngrams(doc, n))
        
        # 计算重叠
        overlap = generated_ngrams.intersection(training_ngrams)
        overlap_ratio = len(overlap) / len(generated_ngrams)
        results[n] = overlap_ratio
    
    return results


def find_longest_common_substring(generated_text: str, training_docs: List[str]) -> Tuple[int, str, str]:
    """
    找到生成文本与训练数据中最长的公共子串
    
    Args:
        generated_text (str): 生成的文本
        training_docs (List[str]): 训练文档列表
        
    Returns:
        Tuple[int, str, str]: (最长长度, 公共子串, 来源文档)
    """
    generated_clean = generated_text.lower()
    max_length = 0
    longest_substring = ""
    source_doc = ""
    
    for doc in training_docs:
        doc_clean = doc.lower()
        
        # 使用SequenceMatcher找到最长公共子序列
        matcher = SequenceMatcher(None, generated_clean, doc_clean)
        match = matcher.find_longest_match(0, len(generated_clean), 0, len(doc_clean))
        
        if match.size > max_length:
            max_length = match.size
            longest_substring = generated_clean[match.a:match.a + match.size]
            source_doc = doc[:200] + "..." if len(doc) > 200 else doc
    
    return max_length, longest_substring, source_doc


def calculate_novelty_score(ngram_overlaps: Dict[int, float], char_overlaps: Dict[int, float],
                           longest_match_ratio: float) -> float:
    """
    计算新颖性得分 (0-1, 1表示完全新颖)
    
    Args:
        ngram_overlaps (Dict[int, float]): 单词n-gram重叠比例
        char_overlaps (Dict[int, float]): 字符n-gram重叠比例
        longest_match_ratio (float): 最长匹配比例
        
    Returns:
        float: 新颖性得分
    """
    # 加权计算重叠分数
    word_score = np.mean(list(ngram_overlaps.values()))
    char_score = np.mean(list(char_overlaps.values()))
    
    # 综合得分 (越低越新颖)
    overlap_score = 0.4 * word_score + 0.4 * char_score + 0.2 * longest_match_ratio
    
    # 转换为新颖性得分
    novelty_score = max(0, 1 - overlap_score)
    
    return novelty_score


def check_text_novelty(generated_text: str, training_data_path: str, 
                      tokenizer_path: str, vocab_filename: str = "vocab.json",
                      merges_filename: str = "merges.txt", max_samples: int = 10000) -> Dict:
    """
    全面检查文本新颖性
    
    Args:
        generated_text (str): 生成的文本
        training_data_path (str): 训练数据路径
        tokenizer_path (str): tokenizer路径
        vocab_filename (str): 词汇表文件名
        merges_filename (str): 合并规则文件名
        max_samples (int): 最大检查的训练样本数
        
    Returns:
        Dict: 检查结果
    """
    print("Loading tokenizer...")
    vocab_path = os.path.join(tokenizer_path, vocab_filename)
    merges_path = os.path.join(tokenizer_path, merges_filename)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    
    print("Loading training data...")
    training_docs = load_training_data(training_data_path, tokenizer, max_samples)
    
    print("Checking exact matches...")
    exact_match, matching_docs = check_exact_match(generated_text, training_docs)
    
    print("Checking n-gram overlaps...")
    ngram_overlaps = check_ngram_overlap(generated_text, training_docs)
    char_overlaps = check_char_ngram_overlap(generated_text, training_docs)
    
    print("Finding longest common substring...")
    longest_len, longest_str, source_doc = find_longest_common_substring(generated_text, training_docs)
    longest_match_ratio = longest_len / len(generated_text) if len(generated_text) > 0 else 0
    
    print("Calculating novelty score...")
    novelty_score = calculate_novelty_score(ngram_overlaps, char_overlaps, longest_match_ratio)
    
    results = {
        "generated_text": generated_text,
        "text_length": len(generated_text),
        "exact_match": exact_match,
        "matching_docs_count": len(matching_docs),
        "ngram_overlaps": ngram_overlaps,
        "char_ngram_overlaps": char_overlaps,
        "longest_common_length": longest_len,
        "longest_common_substring": longest_str,
        "longest_match_ratio": longest_match_ratio,
        "novelty_score": novelty_score,
        "training_samples_checked": len(training_docs)
    }
    
    if matching_docs:
        results["sample_matching_docs"] = matching_docs[:3]  # 只保存前3个匹配文档
    
    return results


def print_novelty_report(results: Dict):
    """打印新颖性检查报告"""
    print("\n" + "="*60)
    print("NOVELTY CHECK REPORT")
    print("="*60)
    
    print(f"Generated text length: {results['text_length']} characters")
    print(f"Training samples checked: {results['training_samples_checked']}")
    
    print(f"\n📊 NOVELTY SCORE: {results['novelty_score']:.3f}")
    if results['novelty_score'] > 0.8:
        print("   ✅ HIGH NOVELTY - Text appears to be genuinely generated")
    elif results['novelty_score'] > 0.5:
        print("   ⚠️  MEDIUM NOVELTY - Some overlap with training data")
    else:
        print("   ❌ LOW NOVELTY - Significant overlap with training data")
    
    print(f"\n🔍 EXACT MATCH CHECK:")
    if results['exact_match']:
        print(f"   ❌ Found {results['matching_docs_count']} exact/substring matches")
        if 'sample_matching_docs' in results:
            print("   Sample matching documents:")
            for i, doc in enumerate(results['sample_matching_docs'][:2]):
                print(f"     {i+1}. {doc[:100]}...")
    else:
        print("   ✅ No exact matches found")
    
    print(f"\n📝 N-GRAM OVERLAP (word-level):")
    for n, overlap in results['ngram_overlaps'].items():
        status = "❌" if overlap > 0.5 else "⚠️" if overlap > 0.2 else "✅"
        print(f"   {status} {n}-gram: {overlap:.3f}")
    
    print(f"\n🔤 CHARACTER N-GRAM OVERLAP:")
    for n, overlap in results['char_ngram_overlaps'].items():
        status = "❌" if overlap > 0.7 else "⚠️" if overlap > 0.4 else "✅"
        print(f"   {status} {n}-char: {overlap:.3f}")
    
    print(f"\n🎯 LONGEST COMMON SUBSTRING:")
    print(f"   Length: {results['longest_common_length']} characters")
    print(f"   Ratio: {results['longest_match_ratio']:.3f}")
    if results['longest_common_length'] > 20:
        print(f"   Content: \n {results['longest_common_substring']}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Check novelty of generated text against training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
                 epilog="""
Example usage:
  # Check a specific generated text
  python check_novelty.py --text "Once upon a time..." --training-data ../tokenized_data/train_tokens.npy --tokenizer-path ../tokenizer_models/TinyStory
  
  # Check text from file
  python check_novelty.py --text-file generated.txt --training-data train.txt --tokenizer-path tokenizer
  
  # Save results to JSON
  python check_novelty.py --text "Hello world" --training-data train_tokens.npy --tokenizer-path tokenizer --output results.json
  
  # Real world example
  python check_novelty.py --text "Once upon a time, there was a little girl named Sue..." --training-data ../tokenized_data/TinyStoriesV2-GPT4-train_tokens.npy --tokenizer-path ../tokenizer_models --vocab-filename tokenizer_vocab_tinystories_10k.json --merges-filename tokenizer_merges_tinystories_10k.txt --max-samples 10000    
        """
    )
    
    # 输入文本
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Generated text to check")
    group.add_argument("--text-file", type=str, dest="text_file", help="File containing generated text")
    
    # 训练数据和tokenizer
    parser.add_argument("--training-data", type=str, required=True, dest="training_data",
                        help="Path to training data (.npy or .txt)")
    parser.add_argument("--tokenizer-path", type=str, required=True, dest="tokenizer_path",
                        help="Path to tokenizer directory")
    parser.add_argument("--vocab-filename", type=str, default="vocab.json", dest="vocab_filename",
                        help="Vocabulary file name")
    parser.add_argument("--merges-filename", type=str, default="merges.txt", dest="merges_filename",
                        help="Merges file name")
    
    # 检查参数
    parser.add_argument("--max-samples", type=int, default=10000, dest="max_samples",
                        help="Maximum training samples to check")
    parser.add_argument("--output", type=str,
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # 读取生成的文本
    if args.text:
        generated_text = args.text
    else:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            generated_text = f.read().strip()
    
    try:
        # 检查新颖性
        results = check_text_novelty(
            generated_text=generated_text,
            training_data_path=args.training_data,
            tokenizer_path=args.tokenizer_path,
            vocab_filename=args.vocab_filename,
            merges_filename=args.merges_filename,
            max_samples=args.max_samples
        )
        
        # 打印报告
        print_novelty_report(results)
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 