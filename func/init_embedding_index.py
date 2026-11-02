try:
    from func import config, utils
    from func.retriever_milvus import Retriever
except Exception as e:
    import config, utils
    from retriever_milvus import Retriever

"""
只负责构建向量数据库
"""


def add_text_metadata(data_list: list) -> list:
    new_data_list = []
    for item in data_list:
        source = item.get("source", {})

        # 如果 source 中某些字段是 list，就用 "-" 拼接
        for k, v in source.items():
            if isinstance(v, list):
                source[k] = "_".join(map(str, v))

        if "category_paths" not in source:
            item['summary'] = f'元数据: {item["company"]}/{item["kind"]}/{item["type"]}/{item["product_name"]}{item["filename"]}\n内容: {item["summary"]}'
        else:
            item['text'] = f'元数据: {source["category_paths"]}{source["filename"]}\n内容: {item["text"]}'
        new_data_list.append(item)

    return new_data_list


def run():
    # corpus_path = f"{config.PROJECT_DIR}/func/data/corpus_cleaned.jsonl"
    corpus_path = f"{config.PROJECT_DIR}/func/sztu_result/result/agent1/result2.json"
    embedding_model = "BAAI/bge-large-zh-v1.5"
    corpus_data_name = utils.get_last_name(corpus_path)

    is_text_add_metadata = False
    force_rebuild_index = True  # 摘要库首次建库用 True

    print("=" * 50)
    print("向量库初始化配置")
    print("=" * 50)
    print(f"语料路径:           {corpus_path}")
    print(f"嵌入模型:           {embedding_model}")
    print(f"是否附加元数据:     {is_text_add_metadata}")
    print(f"是否强制重建索引:   {force_rebuild_index}")
    print("=" * 50)

    corpus_data = utils.load_json(corpus_path)
    if is_text_add_metadata:
        corpus_data = add_text_metadata(corpus_data)
        corpus_data_name = corpus_data_name + "_metadata"

    retriever = Retriever(
        embedding_model,
        collection_name=corpus_data_name,
    )


    if force_rebuild_index:
        retriever.create_collection()

    index_size = retriever.get_index_size()
    if index_size != len(corpus_data):
        import asyncio
        asyncio.run(retriever.add_documents2collection(corpus_data))

    index_size = retriever.get_index_size()
    if index_size == len(corpus_data):
        print("构建成功")
    else:
        print("构建失败")
    res = retriever.bm25_search("保险")
    pass


if __name__ == "__main__":
    run()
