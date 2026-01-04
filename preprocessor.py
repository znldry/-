import json
import re
import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class MedicalTextProcessor:
    """医疗文本处理器：负责清洗、分块和向量化"""
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化处理器
        :param model_name: 嵌入模型名称，推荐的多语言模型
        """
        print(f"[初始化] 加载嵌入模型: {model_name}")
        # 此模型支持中英文，适合你的实验要求
        self.embedding_model = SentenceTransformer(model_name)
        self.model_dimension = self.embedding_model.get_sentence_embedding_dimension()
        print(f"[初始化] 模型维度: {self.model_dimension}")
        
    def load_and_extract_text(self, corpus_path: str) -> str:
        """
        加载原始数据并提取核心文本内容
        :return: 提取出的纯文本字符串
        """
        print(f"[步骤1] 从文件中提取核心文本: {corpus_path}")
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'context' in data:
                main_text = data['context']
                print(f"  ✅ 成功提取 'context' 键下的文本")
                print(f"  📊 文本总长度: {len(main_text)} 字符")
                # 显示开头和结尾部分
                print(f"  🔍 文本开头: {main_text[:100]}...")
                print(f"  🔍 文本结尾: ...{main_text[-100:]}")
                return main_text
            else:
                print("  ❌ 错误：数据字典中未找到 'context' 键")
                return ""
                
        except Exception as e:
            print(f"  ❌ 加载文件时出错: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本：移除多余空格、换行，规范化格式
        """
        print("[步骤2] 清洗文本...")
        # 合并多个换行符和空格
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        # 确保段落之间有适当的空格
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\.\s+', '.\n', text)  # 在句号后重新添加换行，便于识别句子
        print(f"  ✅ 文本清洗完成")
        return text
    
    def split_into_paragraphs(self, text: str, min_paragraph_length: int = 50) -> List[str]:
        """
        将长文本按段落分割（基于换行或句子边界）
        """
        print("[步骤3] 将文本分割为初始段落...")
        # 先按换行分割
        raw_paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # 过滤掉过短的段落（可能是标题或编号）
        paragraphs = [p for p in raw_paragraphs if len(p) >= min_paragraph_length]
        
        print(f"  📊 获得 {len(paragraphs)} 个潜在段落")
        if paragraphs:
            print(f"  🔍 样例段落 (长度: {len(paragraphs[0])}): {paragraphs[0][:100]}...")
        return paragraphs
    
    def chunk_paragraphs(self, paragraphs: List[str], 
                         max_chunk_size: int = 500, 
                         overlap: int = 50) -> List[Dict[str, Any]]:
        """
        智能分块：将段落合并为适当大小的文本块，保持语义完整
        """
        print(f"[步骤4] 智能分块 (目标大小: {max_chunk_size}字符, 重叠: {overlap}字符)...")
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for i, para in enumerate(paragraphs):
            para_length = len(para)
            
            # 如果当前段落本身就很长，需要进一步分割
            if para_length > max_chunk_size:
                # 如果当前块有内容，先保存
                if current_chunk:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk.strip(),
                        'length': len(current_chunk.strip()),
                        'source_paragraphs': f'{i}'
                    })
                    chunk_id += 1
                    current_chunk = ""
                    current_length = 0
                
                # 对长段落按句子分割
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub_chunk = ""
                for sent in sentences:
                    sent_length = len(sent)
                    if len(sub_chunk) + sent_length <= max_chunk_size:
                        sub_chunk += " " + sent if sub_chunk else sent
                    else:
                        if sub_chunk:
                            chunks.append({
                                'chunk_id': chunk_id,
                                'text': sub_chunk.strip(),
                                'length': len(sub_chunk.strip()),
                                'source_paragraphs': f'{i}(部分)'
                            })
                            chunk_id += 1
                        sub_chunk = sent
                
                # 处理剩余部分
                if sub_chunk:
                    current_chunk = sub_chunk
                    current_length = len(sub_chunk)
            
            # 如果段落可以加入当前块
            elif current_length + para_length <= max_chunk_size:
                current_chunk += " " + para if current_chunk else para
                current_length += para_length
            
            # 如果段落太大，结束当前块并开始新块
            else:
                if current_chunk:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk.strip(),
                        'length': len(current_chunk.strip()),
                        'source_paragraphs': f'{i-1 if i>0 else i}'
                    })
                    chunk_id += 1
                
                # 新块从当前段落开始，并包含一些重叠
                if overlap > 0 and current_chunk:
                    # 从上一块取末尾部分作为重叠
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
                current_length = len(current_chunk)
        
        # 处理最后一个块
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk.strip()),
                'source_paragraphs': 'final'
            })
        
        print(f"  ✅ 生成 {len(chunks)} 个文本块")
        print(f"  📊 块大小统计:")
        if chunks:
            lengths = [c['length'] for c in chunks]
            print(f"    最小: {min(lengths)} 字符, 最大: {max(lengths)} 字符, 平均: {sum(lengths)/len(lengths):.1f} 字符")
        
        # 显示前3个块作为样例
        for i, chunk in enumerate(chunks[:3]):
            print(f"  🔍 块{i+1} (长度:{chunk['length']}): {chunk['text'][:80]}...")
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        为所有文本块生成嵌入向量
        """
        print(f"[步骤5] 为 {len(chunks)} 个文本块生成嵌入向量...")
        
        texts = [chunk['text'] for chunk in chunks]
        
        # 批量生成嵌入（显示进度条）
        print("  正在编码... (这可能需要一些时间)")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        print(f"  ✅ 嵌入生成完成")
        print(f"  📊 嵌入形状: {embeddings.shape}")
        
        # 更新块信息，添加嵌入向量
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        return embeddings, chunks
    
    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], output_path: str):
        """将处理后的块保存为JSON文件（用于调试和检查）"""
        # 注意：嵌入向量很大，保存时转换为列表
        save_data = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            if 'embedding' in chunk_copy:
                chunk_copy['embedding'] = chunk_copy['embedding'].tolist() if hasattr(chunk_copy['embedding'], 'tolist') else list(chunk_copy['embedding'])
            save_data.append(chunk_copy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 块数据已保存到: {output_path}")
    
    def process_pipeline(self, corpus_path: str, output_dir: str = "processed_data") -> List[Dict[str, Any]]:
        """
        完整的处理流水线
        """
        print("=" * 60)
        print("开始医疗文本处理流水线")
        print("=" * 60)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 加载并提取文本
        raw_text = self.load_and_extract_text(corpus_path)
        if not raw_text:
            print("❌ 无法提取文本，流水线终止")
            return []
        
        # 2. 清洗文本
        cleaned_text = self.clean_text(raw_text)
        
        # 3. 分割段落
        paragraphs = self.split_into_paragraphs(cleaned_text)
        
        # 4. 智能分块
        chunks = self.chunk_paragraphs(paragraphs, max_chunk_size=600, overlap=80)
        
        # 5. 生成嵌入向量
        embeddings, chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # 6. 保存结果
        output_path = os.path.join(output_dir, "medical_chunks.json")
        self.save_chunks_to_json(chunks_with_embeddings, output_path)
        
        print("=" * 60)
        print("✅ 文本处理流水线完成！")
        print(f"   生成 {len(chunks_with_embeddings)} 个文本块")
        print(f"   嵌入维度: {self.model_dimension}")
        print("=" * 60)
        
        return chunks_with_embeddings

def main():
    """主函数"""
    
    # 1. 初始化处理器
    processor = MedicalTextProcessor()
    
    # 2. 定义路径（根据你的实际路径调整）
    base_path = "D:/lesson/exp4/GraphRAG-Benchmark-main"
    corpus_path = os.path.join(base_path, "Data", "Corpus", "medical.json").replace('\\', '/')
    output_dir = os.path.join(base_path, "processed_data").replace('\\', '/')
    
    # 3. 运行完整流水线
    processed_chunks = processor.process_pipeline(corpus_path, output_dir)
    
    # 4. 给出下一步建议
    if processed_chunks:
        print("\n【下一步建议】")
        print("1. 检查生成的文本块文件: processed_data/medical_chunks.json")
        print("2. 准备构建向量数据库 (Milvus)")
        print("   运行以下命令:")
        print(f"   cd {base_path}")
        print("   python src/vector_store.py  # 我将为你创建此文件")
        print("\n3. 测试检索功能")
        print("   使用 medical_questions.json 中的问题进行检索测试")

if __name__ == "__main__":
    # Windows控制台编码设置
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    main()