import re

def _extract_index_info(content):
    info = {
        "importance": "★★★☆☆",
        "conclusion": "要約参照",
        "practical": "要約参照"
    }
    
    # サマリーインデックス情報セクションを探す
    section_match = re.search(r"(?:##+|[*]{2})\s*サマリーインデックス情報.*?\n(.*?)(?=\n#|$)", content, re.DOTALL | re.IGNORECASE)
    
    section_content = section_match.group(1) if section_match else content
    print("--- SECTION CONTENT ---")
    print(repr(section_content))
        
    imp_match = re.search(r"重要度.*?([★☆]+)", section_content)
    if imp_match:
        info["importance"] = imp_match.group(1).strip()
            
    conc_match = re.search(r"結論(.*?)(?:実用|$)", section_content, re.DOTALL)
    if conc_match:
        val = conc_match.group(1)
        print("CONC RAW:", repr(val))
        val = re.sub(r"^[:：\s\*・-]*", "", val)
        val = re.sub(r"[\s\*・\-\d\.]*$", "", val)
        print("CONC CLEAN:", repr(val))
        if val:
            info["conclusion"] = val
            
    prac_match = re.search(r"実用(.*?)$", section_content, re.DOTALL)
    if prac_match:
        val = prac_match.group(1)
        print("PRAC RAW:", repr(val))
        val = re.sub(r"^[:：\s\*・-]*", "", val)
        val = re.sub(r"[\s\*・\-\d\.]*$", "", val)
        print("PRAC CLEAN:", repr(val))
        if val:
            info["practical"] = val
            
    return info

papers = [
    ("Pattern A", "## サマリーインデックス情報\n- 重要度: ★★★★★\n- 結論: 正常に抽出されるはずです。\n- 実用: この機能は**非常に重要**です。\n\n## まず一言で"),
    ("Pattern B", "## サマリーインデックス情報\n**重要度**: ★★★★☆\n**結論**: \n複数行にわたる内容も\n正しく取得できるかテスト。\n\n**実用**: **明日からの診療**に役立ちます。\n\n# 次のセクション"),
    ("Pattern C", "## サマリーインデックス情報\n1. 重要度：★★☆☆☆\n2. 結論：数字リスト形式でも問題なし\n3. 実用：安定して動作します\n##"),
    ("Pattern D", "## サマリーインデックス情報\n重要度 ★★★☆☆\n結論 ラベル直後の空白だけでもパース \n実用 柔軟性を追求しました")
]

for title, p in papers:
    print(f"\n[{title}]")
    print(_extract_index_info(p))

