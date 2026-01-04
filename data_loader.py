import json
import os
import sys

class MedicalDataLoader:
    """医疗RAG系统数据加载器（兼容字典格式数据）"""
    
    def __init__(self, base_path="D:/lesson/exp4/GraphRAG-Benchmark-main"):
        self.base_path = base_path.replace('\\', '/')
        self.corpus_path = os.path.join(self.base_path, "Data", "Corpus", "medical.json").replace('\\', '/')
        self.questions_path = os.path.join(self.base_path, "Data", "Questions", "medical_questions.json").replace('\\', '/')
        
    def load_corpus(self, sample_num=3):
        """加载医疗知识库文档（兼容列表或字典格式）"""
        print(f"[1] 正在加载知识库文件: {self.corpus_path}")
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            # 首先确定数据结构
            data_type = type(corpus_data).__name__
            print(f"    数据结构: {data_type}")
            
            # 处理字典格式的数据
            if isinstance(corpus_data, dict):
                print(f"    ✅ 成功加载 {len(corpus_data)} 条医疗文档（字典格式）")
                doc_items = list(corpus_data.items())
                
                print(f"\n    【知识库样本（前{min(sample_num, len(doc_items))}条）】")
                for i, (doc_id, doc_content) in enumerate(doc_items[:sample_num]):
                    print(f"    --- 文档 {i+1} (ID: {doc_id}) ---")
                    
                    # 检查doc_content本身是字符串还是字典
                    if isinstance(doc_content, dict):
                        # 如果是字典，打印所有键值对
                        for key, value in doc_content.items():
                            if key == 'content' or key == 'text':
                                preview = str(value)[:150] + "..." if len(str(value)) > 150 else str(value)
                                print(f"      {key}: {preview}")
                            else:
                                print(f"      {key}: {value}")
                    else:
                        # 如果是字符串，直接显示预览
                        preview = str(doc_content)[:150] + "..." if len(str(doc_content)) > 150 else str(doc_content)
                        print(f"      内容预览: {preview}")
                    print()
                
                return corpus_data, 'dict'
                
            # 处理列表格式的数据
            elif isinstance(corpus_data, list):
                print(f"    ✅ 成功加载 {len(corpus_data)} 条医疗文档（列表格式）")
                
                print(f"\n    【知识库样本（前{min(sample_num, len(corpus_data))}条）】")
                for i, doc in enumerate(corpus_data[:sample_num]):
                    print(f"    --- 文档 {i+1} ---")
                    if isinstance(doc, dict):
                        for key, value in doc.items():
                            if key == 'content' or key == 'text':
                                preview = str(value)[:150] + "..." if len(str(value)) > 150 else str(value)
                                print(f"      {key}: {preview}")
                            else:
                                print(f"      {key}: {value}")
                    else:
                        print(f"      内容: {str(doc)[:150]}...")
                    print()
                
                return corpus_data, 'list'
            else:
                print(f"    ❌ 未知的数据结构: {data_type}")
                return None, None
                
        except FileNotFoundError:
            print(f"    ❌ 错误：找不到文件 {self.corpus_path}")
            return None, None
        except Exception as e:
            print(f"    ❌ 加载时发生错误: {e}")
            return None, None
    
    def load_questions(self, sample_num=3):
        """加载测试问题集（同样兼容多种格式）"""
        print(f"[2] 正在加载问题文件: {self.questions_path}")
        try:
            with open(self.questions_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            data_type = type(questions_data).__name__
            print(f"    数据结构: {data_type}")
            
            if isinstance(questions_data, dict):
                print(f"    ✅ 成功加载 {len(questions_data)} 个测试问题（字典格式）")
                q_items = list(questions_data.items())
                
                print(f"\n    【测试问题样本（前{min(sample_num, len(q_items))}个）】")
                for i, (q_id, q_content) in enumerate(q_items[:sample_num]):
                    print(f"    问题 {i+1} (ID: {q_id})")
                    
                    if isinstance(q_content, dict):
                        question_text = q_content.get('question', 'N/A')
                        answer = q_content.get('answer', 'N/A')
                    else:
                        question_text = str(q_content)
                        answer = 'N/A'
                    
                    print(f"      问题: {question_text}")
                    if answer != 'N/A':
                        ans_preview = str(answer)[:100] + "..." if len(str(answer)) > 100 else str(answer)
                        print(f"      答案预览: {ans_preview}")
                    print()
                    
            elif isinstance(questions_data, list):
                print(f"    ✅ 成功加载 {len(questions_data)} 个测试问题（列表格式）")
                
                print(f"\n    【测试问题样本（前{min(sample_num, len(questions_data))}个）】")
                for i, q in enumerate(questions_data[:sample_num]):
                    print(f"    问题 {i+1}")
                    
                    if isinstance(q, dict):
                        question_text = q.get('question', 'N/A')
                        answer = q.get('answer', 'N/A')
                    else:
                        question_text = str(q)
                        answer = 'N/A'
                    
                    print(f"      问题: {question_text}")
                    if answer != 'N/A':
                        ans_preview = str(answer)[:100] + "..." if len(str(answer)) > 100 else str(answer)
                        print(f"      答案预览: {ans_preview}")
                    print()
            else:
                print(f"    ❌ 未知的数据结构: {data_type}")
                
            return questions_data
            
        except Exception as e:
            print(f"    ❌ 加载问题时出错: {e}")
            return None
    
    def analyze_content_type(self, corpus_data, data_format):
        """分析内容类型（HTML vs 纯文本）"""
        print("[3] 分析内容类型...")
        
        sample_text = ""
        
        # 根据数据结构获取样本文本
        if data_format == 'dict' and corpus_data:
            first_item = next(iter(corpus_data.values()))
            if isinstance(first_item, dict):
                sample_text = first_item.get('content', first_item.get('text', ''))
            else:
                sample_text = str(first_item)
        elif data_format == 'list' and corpus_data:
            first_item = corpus_data[0]
            if isinstance(first_item, dict):
                sample_text = first_item.get('content', first_item.get('text', ''))
            else:
                sample_text = str(first_item)
        
        # 判断内容类型
        if sample_text:
            # 检查是否是HTML
            is_html = sample_text.strip().startswith('<') and '>' in sample_text
            
            # 检查常见HTML标签
            html_tags = ['<p>', '<div>', '<html>', '<body>', '<h1>', '<br>']
            has_html_tags = any(tag in sample_text.lower() for tag in html_tags)
            
            if is_html or has_html_tags:
                content_type = "HTML"
                print(f"    ✅ 内容类型: {content_type}")
                print(f"    📄 样本开头: {sample_text[:100]}...")
            else:
                content_type = "纯文本"
                print(f"    ✅ 内容类型: {content_type}")
                print(f"    📄 样本开头: {sample_text[:100]}...")
            
            return content_type
        else:
            print("    ⚠️  无法确定内容类型")
            return "未知"

def main():
    """主函数：执行数据加载演示"""
    print("=" * 60)
    print("医疗RAG系统 - 数据加载与验证（兼容版）")
    print("=" * 60)
    
    # 创建加载器实例
    loader = MedicalDataLoader()
    
    # 1. 加载知识库
    corpus, corpus_format = loader.load_corpus()
    
    # 2. 分析内容类型
    content_type = None
    if corpus:
        content_type = loader.analyze_content_type(corpus, corpus_format)
    
    # 3. 加载问题集
    questions = loader.load_questions()
    
    print("\n" + "=" * 60)
    if corpus and questions:
        print("✅ 数据加载验证完成！")
        print("\n【关键发现】")
        print(f"  1. 知识库格式: {corpus_format}")
        print(f"  2. 内容类型: {content_type}")
        print(f"  3. 文档数量: {len(corpus) if corpus else 0}")
        print(f"  4. 问题数量: {len(questions) if isinstance(questions, (list, dict)) else '未知'}")
        return True, content_type, corpus_format
    else:
        print("❌ 数据加载存在问题，请检查上述错误。")
        return False, None, None

if __name__ == "__main__":
    # 设置控制台编码为UTF-8，确保中文正常显示
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    success, content_type, data_format = main()
    
    # 给出下一步提示
    if success:
        print("\n" + "=" * 60)
        print("【下一步行动计划】")
        
        if content_type == "HTML":
            print("1. 创建 HTML 预处理模块 (preprocessor.py)")
            print("   - 使用 BeautifulSoup 清理 HTML 标签")
            print("   - 提取纯文本内容")
            print("   - 保留重要结构信息（如标题、段落）")
        elif content_type == "纯文本":
            print("1. 创建文本预处理模块 (preprocessor.py)")
            print("   - 文本清洗（去除多余空白、特殊字符）")
            print("   - 中文句子分割")
        
        print("2. 设计文本分块策略")
        print("   - 确定合适的块大小（如500字符）")
        print("   - 设置块重叠（如50字符）")
        print("   - 保持语义完整性")
        
        print("3. 开始构建向量数据库")
        print("   - 选择嵌入模型（如 text2vec 中文模型）")
        print("   - 将文本块转换为向量")
        print("   - 存入 Milvus 数据库")
        
        print("\n运行以下命令开始预处理：")
        print("  # 我将为你创建 preprocessor.py 文件")
        print("  python src/preprocessor.py")