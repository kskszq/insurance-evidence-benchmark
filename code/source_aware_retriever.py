import asyncio

from chatbot import ChatModel
from query_metadata import predict_query_metadata
from retriever_milvus import Reranker, Retriever

llm = ChatModel()

"""
ours_agent  - 我们自己的方法
"""


def chunk_relevance(query, chunk_list):
    # 格式化检索块，带上下标
    chunks_formatted = "<chunks>\n" + "\n".join(
        [
            f'  <chunk index="{i}">\n'
            f'    <path>{chunk.get("category_paths", "")}</path>\n'
            f'    <text>{chunk.get("text", "")[:1500]}</text>\n'
            f'  </chunk>'
            for i, chunk in enumerate(chunk_list)
        ]
    ) + "\n</chunks>"

    prompt = f"""\
# 角色
你是一个专业的文档检索评估与查询优化专家。你的核心任务是评估给定的 XML 格式检索文本块（Chunks）是否能够为回答用户的查询（Query）提供**实质性支撑**，并在必要时启动双流自适应检索机制，严格围绕用户原意对查询进行精准重写。

# 工作流
1. **深度分析与相关性评估**：
- 首先阅读 Query，精准拆解用户的核心意图和关键诉求。
- 逐一评估 `<chunk>` 的 `<text>` 内容。
- **核心判定标准**：该块是否包含了回答 Query 所需的关键事实、数据、业务逻辑或重要上下文？

2. **有效索引提取**：
- 基于上述分析，记录所有能为解答查询提供帮助的 `<chunk>` 的 `index` 属性值。

3. **双流自适应查询重写**：
- 如果提取到的有效 Chunk 为空，说明当前检索未能覆盖原意图，必须触发重写机制。重写必须**严格忠于用户原始诉求**，禁止过度联想与发散：
- **BM25 视角重写**：精准提取 Query 中的核心实体、专有名词、核心动作和数值条件，剔除无意义的口语和停用词，生成适合精确匹配的关键词组（以空格分隔），**确保不偏离原问题焦点**。
- **Vector 视角重写**：将 Query 转化为语义清晰、结构完整的自然语言问句或陈述句。在**保持原始查询核心意图不变**的前提下，优化表达结构，使其更适合密集向量（Dense Vector）的语义相似度匹配。

4. **格式化输出**：
- 严格且仅输出符合下述结构的 JSON，严禁包含任何其他正文说明、思考过程或 Markdown 代码块外的文字。

# 输出格式
{{
"reasoning": "简明扼要地说明判定逻辑。例如：'Chunk 0提供了部分指标，Chunk 2补充了规则，均予以采纳；Chunk 1为无关背景。' 或 '所有Chunk均偏离原问题焦点，需触发重写。'",
"relevant_indices": [0, 2],
"rewrite_needed": false,
"bm25_rewrite": "BM25视角的关键词，若无需重写则输出 null",
"vector_rewrite": "Vector视角的语义重写句子，若无需重写则输出 null"
}}

# 输入
- Query: {query}
- Chunks: 
=========
{chunks_formatted}
========="""
    message = [
        {
            'role': 'user',
            'content': prompt,
        }
    ]
    res = asyncio.run(llm.async_chat(message, is_response_json=True))
    return res


not_relevant_nums = 0

from collections import defaultdict
from typing import List, Dict


class HybridRAGFuser:
    def __init__(self, k: int = 60, doc_weight: float = 0.3):
        """
        :param k: RRF 公式中的常数，通常取 60
        :param doc_weight: 文档名称检索通道的权重占比 (0.0 ~ 1.0)
        """
        self.k = k
        self.doc_weight = doc_weight

    def fuse(
            self,
            bm25_chunks: List[Dict],
            vector_chunks: List[Dict],
            doc_name_results: List[str]
    ) -> List[Dict]:
        """
        三通道融合
        :param bm25_chunks: BM25 返回的块列表，带 'id' 和 'file_name'
        :param vector_chunks: 向量返回的块列表，带 'id' 和 'file_name'
        :param doc_name_results: 文档名称通道返回的文档名排序列表 (如 ["doc_A", "doc_B"])
        """
        fused_scores = defaultdict(float)
        chunk_store = {}

        # --- 步骤 1: 处理块级别的双通道 (BM25 + Vector) ---
        # 权重分配：块级内部通常平分 (1 - doc_weight)
        chunk_channel_weight = (1.0 - self.doc_weight) / 2.0

        for results in [bm25_chunks, vector_chunks]:
            for rank, chunk in enumerate(results, start=1):
                chunk_id = chunk["id"]
                chunk_store[chunk_id] = chunk
                # 累加块级 RRF 分数
                fused_scores[chunk_id] += chunk_channel_weight * (1.0 / (self.k + rank))

        # --- 步骤 2: 处理文档级别的全局加权 (Doc Name) ---
        # 将文档排名转换为 Boost 映射表
        doc_boost_map = {
            doc_name: 1.0 / (self.k + rank)
            for rank, doc_name in enumerate(doc_name_results, start=1)
        }

        # --- 步骤 3: 融合计算 ---
        final_results = []
        for chunk_id, base_score in fused_scores.items():
            chunk = chunk_store[chunk_id]
            file_name = chunk.get("filename")  # 假设块数据中有所属文件名

            # 获取该块所属文档的全局得分
            # 如果该文档不在文档检索的前 N 名中，doc_boost 为 0
            doc_boost = doc_boost_map.get(file_name, 0.0)

            # 最终分数 = 块级混合分 + 文档级权重分
            total_score = base_score + (self.doc_weight * doc_boost)

            # 记录详细分数便于调试
            chunk["rrf_score"] = total_score
            chunk["doc_boost_contribution"] = self.doc_weight * doc_boost
            final_results.append(chunk)

        # 排序返回
        return sorted(final_results, key=lambda x: x["rrf_score"], reverse=True)


from concurrent.futures import ThreadPoolExecutor


def new_try(
        qa_item: dict,
        corpus_retriever: Retriever,
        summary_retriever: Retriever,
        reranker: Reranker = None,
):
    question = qa_item["question"]

    query_metadata = predict_query_metadata(question)
    predict_company = query_metadata["predictions"].get("company")
    filter = {"company": predict_company} if predict_company else None

    def search(query, f, w, bm25_rewrite=None):
        if bm25_rewrite is None:
            bm25_rewrite = query
        with ThreadPoolExecutor(max_workers=3) as executor:
            dense_future = executor.submit(
                corpus_retriever.embedding_search,
                query=query,
                search_num=500,
                filter=f
            )

            sparse_future = executor.submit(
                corpus_retriever.bm25_search,
                query=bm25_rewrite,
                search_num=500,
                filter=f
            )

            def run_async(coro):
                return asyncio.run(coro)

            summary_future = executor.submit(
                run_async,
                summary_retriever.hybrid_search(
                    query=query,
                    search_num=500,
                    filter=f
                )
            )

            vec_res = dense_future.result()
            bm25_res = sparse_future.result()

            summary_list = summary_future.result()
            fuser = HybridRAGFuser(doc_weight=w)
            file_name_list = [item['filename'] for item in summary_list]
            final_top_chunks = fuser.fuse(bm25_res, vec_res, file_name_list)
            rerank_res = reranker.rerank_docs(query, final_top_chunks)
            return rerank_res, summary_list

    rerank_res, summary_list = search(question, filter, 0.5)
    not_relevant_chunks = []
    top_10_chunks = rerank_res[:10]
    # Verify retrieved content for every query; retry only when no chunk is relevant.
    max_retry = 3
    retry_count = 0

    while retry_count < max_retry:
        try:
            chunk_relevance_res = chunk_relevance(question, top_10_chunks)
        except Exception as error:
            # LLM verification is an optional final-stage refinement. A
            # temporary gateway failure must not discard valid retrieval and
            # reranker results for the whole QA sample.
            print(
                "[new_try] LLM verification unavailable; "
                f"keeping current retrieval results. error: {error}"
            )
            break
        relevant_indices = chunk_relevance_res['relevant_indices']

        if isinstance(relevant_indices, list) and len(relevant_indices) != 0:
            # ✅ 块相关，跳出循环
            break
        else:
            # ❌ 块不相关，进行二次检索
            not_relevant_chunks.extend(top_10_chunks)
            global not_relevant_nums
            not_relevant_nums += 1
            print(f"not relevant nums: {not_relevant_nums}, retry: {retry_count + 1}")
            bm25_rewrite = chunk_relevance_res['bm25_rewrite']
            vector_rewrite = chunk_relevance_res['vector_rewrite']
            filter = {"company": predict_company} if predict_company else None
            res = asyncio.run(corpus_retriever.hybrid_search(
                vector_rewrite,
                filter=filter,
                bm25_query=bm25_rewrite,
            ))
            re_search_results = reranker.rerank_docs(question, res)

            unique_ids = {item["id"] for item in not_relevant_chunks}
            rerank_res = [
                item for item in re_search_results
                if item["id"] not in unique_ids
            ]

            top_10_chunks = rerank_res[:10]

            retry_count += 1

        # 超过最大重试次数仍不相关
        if retry_count == max_retry and (
                not isinstance(relevant_indices, list) or len(relevant_indices) == 0
        ):
            print("Max retry reached, still not relevant.")
    return {
        'search_results': rerank_res,
        'summary_results': summary_list,
        'query_metadata': query_metadata,
    }


def new_try_no_verify(
        qa_item: dict,
        corpus_retriever: Retriever,
        summary_retriever: Retriever,
        reranker: Reranker = None,
):
    question = qa_item["question"]

    query_metadata = predict_query_metadata(question)
    predict_company = query_metadata["predictions"].get("company")
    filter = {"company": predict_company} if predict_company else None

    def search(query, f, w, bm25_rewrite=None):
        if bm25_rewrite is None:
            bm25_rewrite = query
        with ThreadPoolExecutor(max_workers=3) as executor:
            dense_future = executor.submit(
                corpus_retriever.embedding_search,
                query=query,
                search_num=500,
                filter=f
            )

            sparse_future = executor.submit(
                corpus_retriever.bm25_search,
                query=bm25_rewrite,
                search_num=500,
                filter=f
            )

            def run_async(coro):
                return asyncio.run(coro)

            summary_future = executor.submit(
                run_async,
                summary_retriever.hybrid_search(
                    query=query,
                    search_num=500,
                    filter=f
                )
            )

            vec_res = dense_future.result()
            bm25_res = sparse_future.result()

            summary_list = summary_future.result()
            fuser = HybridRAGFuser(doc_weight=w)
            file_name_list = [item['filename'] for item in summary_list]
            final_top_chunks = fuser.fuse(bm25_res, vec_res, file_name_list)
            rerank_res = reranker.rerank_docs(query, final_top_chunks)
            return rerank_res, summary_list

    rerank_res, summary_list = search(question, filter, 0.5)
    return {
        'search_results': rerank_res,
        'summary_results': summary_list,
        'query_metadata': query_metadata,
    }


def summary_company_hybrid__hybrid_rerank(
        qa_item: dict,
        corpus_retriever: Retriever,
        summary_retriever: Retriever,
        reranker: Reranker = None,
):
    question = qa_item["question"]

    query_metadata = predict_query_metadata(question)
    predict_company = query_metadata["predictions"].get("company")
    filter = {"company": predict_company} if predict_company else None

    summary_list = asyncio.run(summary_retriever.hybrid_search(
        question,
        filter=filter,
    ))
    file_name_list = [item['filename'] for item in summary_list]

    filter = {'filename': file_name_list[:50]}
    if predict_company:
        filter['company'] = predict_company
    res = asyncio.run(corpus_retriever.hybrid_search(
        question,
        filter=filter,
    ))
    rerank_res = reranker.rerank_docs(question, res)
    return {
        'search_results': rerank_res,
        'summary_results': summary_list,
        'query_metadata': query_metadata,
    }


def summary_company_hybrid_rerank__hybrid_rerank(
        qa_item: dict,
        corpus_retriever: Retriever,
        summary_retriever: Retriever,
        reranker: Reranker = None,
):
    question = qa_item["question"]

    query_metadata = predict_query_metadata(question)
    predict_company = query_metadata["predictions"].get("company")
    filter = {"company": predict_company} if predict_company else None

    summary_list = asyncio.run(summary_retriever.hybrid_search(
        question,
        filter=filter,
    ))
    summary_list = reranker.rerank_docs(question, summary_list)
    file_name_list = [item['filename'] for item in summary_list]

    filter = {'filename': file_name_list[:100]}
    if predict_company:
        filter['company'] = predict_company
    res = asyncio.run(corpus_retriever.hybrid_search(
        question,
        filter=filter,
    ))
    rerank_res = reranker.rerank_docs(question, res)
    return {
        'search_results': rerank_res,
        'summary_results': summary_list,
        'query_metadata': query_metadata,
    }
