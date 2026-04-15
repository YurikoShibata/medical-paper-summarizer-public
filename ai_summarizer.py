"""
AI要約モジュール

Google Gemini APIを使用して、各論文の批判的要約を
日本語で生成する。モデルフォールバックチェーン対応。
"""

import time
import logging
from typing import Optional

import google.generativeai as genai

from pubmed_searcher import Paper

logger = logging.getLogger(__name__)


class AISummarizer:
    """AI論文要約クラス"""

    def __init__(self, config: dict, api_key: str):
        """
        初期化

        Args:
            config: config.yamlから読み込んだ設定辞書
            api_key: Gemini APIキー
        """
        self.config = config
        self.ai_config = config.get("ai", {})
        self.model_chain = self.ai_config.get("model_chain", [
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite-preview-09-2025",
            "gemini-2.0-flash"
        ])
        self.max_retries = self.ai_config.get("max_retries", 3)
        self.retry_delay = self.ai_config.get("retry_delay", 5)
        self.timeout = self.ai_config.get("timeout", 120)

        genai.configure(api_key=api_key)

    def _create_model(self, model_name: str):
        """指定モデルのインスタンスを作成する"""
        return genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.3,  # 正確性重視で低めの温度
                max_output_tokens=4096,
            )
        )

    def _call_with_fallback(self, prompt: str) -> Optional[str]:
        """
        フォールバックチェーンでAPIを呼び出す

        上位モデルからエラー/レート制限時に下位モデルへ順にフォールバック

        Args:
            prompt: 送信するプロンプト

        Returns:
            生成されたテキスト（全モデル失敗時はNone）
        """
        for model_name in self.model_chain:
            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"モデル {model_name} を使用中"
                        f"（試行 {attempt + 1}/{self.max_retries}）"
                    )
                    model = self._create_model(model_name)
                    # タイムアウトを設定して呼び出し
                    response = model.generate_content(
                        prompt, 
                        request_options={"timeout": self.timeout}
                    )

                    if response.text:
                        logger.info(f"[OK] {model_name} で生成成功")
                        return response.text

                except Exception as e:
                    error_msg = str(e).lower()
                    logger.warning(
                        f"モデル {model_name} でエラー "
                        f"（試行 {attempt + 1}）: {e}"
                    )

                    # レート制限やモデル未対応の場合は次のモデルへ
                    if any(kw in error_msg for kw in [
                        "rate limit", "quota", "429",
                        "not found", "404", "not supported",
                        "504", "deadline", "timeout"
                    ]):
                        logger.info(f"→ 次のモデルに早めにフォールバックします")
                        break

                    # その他のエラーはリトライ
                    if attempt < self.max_retries - 1:
                        wait = self.retry_delay * (attempt + 1)
                        logger.info(f"  {wait}秒後にリトライ...")
                        time.sleep(wait)

        logger.error("全モデルで生成に失敗しました")
        return None

    def summarize_papers(
        self, papers: list[Paper], detailed_top_n: int = 10
    ) -> list[Paper]:
        """
        論文リストの要約を生成する

        Args:
            papers: 優先度順にソートされた論文リスト
            detailed_top_n: 詳細要約を行う上位論文数

        Returns:
            要約が付与された論文リスト
        """
        if detailed_top_n is None:
            detailed_top_n = self.config.get("search", {}).get(
                "detailed_top_n", 3
            )

        total = len(papers)
        logger.info(
            f"{total}件の論文を要約します"
            f"（詳細: {min(detailed_top_n, total)}件）"
        )

        for i, paper in enumerate(papers):
            is_detailed = (i < detailed_top_n)
            mode_str = "詳細" if is_detailed else "簡潔"

            logger.info(
                f"[{i+1}/{total}] {mode_str}要約中: "
                f"{paper.title[:50]}..."
            )

            prompt = self._build_prompt(paper, is_detailed)
            result = self._call_with_fallback(prompt)

            if result:
                paper.summary = {
                    "mode": "detailed" if is_detailed else "brief",
                    "content": result
                }
            else:
                paper.summary = {
                    "mode": "detailed" if is_detailed else "brief",
                    "content": "⚠ 要約の生成に失敗しました。"
                }

            # API呼び出し間隔を空ける
            if i < total - 1:
                time.sleep(2)

        return papers

    def _build_prompt(self, paper: Paper, detailed: bool) -> str:
        """
        論文要約プロンプトを構築する

        Args:
            paper: 論文データ
            detailed: 詳細要約かどうか

        Returns:
            プロンプト文字列
        """
        # 著者表示（最大5名 + et al.）
        if len(paper.authors) > 5:
            author_str = ", ".join(paper.authors[:5]) + " et al."
        else:
            author_str = ", ".join(paper.authors)

        # 論文タイプ
        pub_type_str = ", ".join(paper.pub_types) if paper.pub_types else "不明"

        # 基本情報
        paper_info = f"""
【論文情報】
タイトル: {paper.title}
著者: {author_str}
ジャーナル: {paper.journal}
出版日: {paper.pub_date}
論文タイプ: {pub_type_str}
DOI: {paper.doi if paper.doi else "N/A"}
MeSH用語: {", ".join(paper.mesh_terms[:10]) if paper.mesh_terms else "N/A"}

【アブストラクト】
{paper.abstract}
""".strip()

        if detailed:
            return self._build_detailed_prompt(paper_info)
        else:
            return self._build_brief_prompt(paper_info)

    def _build_detailed_prompt(self, paper_info: str) -> str:
        """詳細要約プロンプト"""
        return f"""あなたは循環器内科の専門医であり、優秀な医師アシスタントです。
以下の論文について、忙しい循環器内科医が短時間で本質をつかめる形式で、日本語で詳細な批判的要約を作成してください。

{paper_info}

以下の形式で出力してください。各セクションは明確に分けてください。
※「承知いたしました」「要約します」等といった前置きや挨拶は一切含めず、いきなり「## サマリーインデックス情報」から出力してください。

## サマリーインデックス情報
冒頭にインデックスを作成するため、以下の3点を極めて簡潔に出力してください。
- **重要度**: [重要度の基準]に従い、★を5つ並べて表記（例：★★★★★）
- **結論**: [40文字以内]で、この論文が何を示したか
- **実用**: [50文字以内]で、明日の臨床にどう活きるか。疾患名、薬剤名、具体的な数値などの重要キーワードは必ず**太字**にすること

## まず一言で
この論文が何を示したのかを1〜2文で日本語要約してください。

## 研究の概要
- **研究背景**: なぜこの研究が行われたか
- **研究デザイン**: どのような研究手法か（RCT、コホート等）
- **対象患者**: どのような患者が対象か（人数・特徴）
- **介入/比較**: 何を比較したか
- **主要評価項目**: 何を評価したか
- **主な結果**: 主要な数値結果（ハザード比、95%CI等を含む）

## 臨床的に重要なポイント
- 専門医の視点で何が重要か
- どの患者で役立つか
- 実臨床を変える可能性があるか
- 現場でどう使うか

## 限界
- バイアスの可能性
- 一般化可能性の限界
- サンプルサイズの問題
- 観察研究であることの限界（該当する場合）
- 対象患者の偏り
- 実装上の課題

## 日本の臨床への実践メモ
- 明日からの診療で意識すべきこと
- カンファレンスで紹介するなら何を強調するか
- 患者説明やチーム共有にどう活かせるか
- 日本の医療環境での適用可能性

重要度の基準：
★★★★★：明日の診療方針に直結する、必ず読むべきパラダイムシフト
★★★★☆：実用性が高く、知っておくべき重要な知見
★★★☆☆：特定の条件下で役立つ、または興味深い知見
★★☆☆☆：参考程度
★☆☆☆☆：現在の業務への直接的な影響は少ない

重要な注意事項:
- 不確かなことは断定しないでください
- 抄録の内容をなぞるだけでなく、批判的吟味を加えてください
- 根拠が弱い場合は弱いと明確に述べてください
- 誇張表現は避けてください
- 統計の細かい説明よりも臨床的解釈を優先してください
- ただし結果の信頼性に関わる統計上の注意点は簡潔に述べてください
"""

    def _build_brief_prompt(self, paper_info: str) -> str:
        """簡潔要約プロンプト"""
        return f"""あなたは循環器内科の専門医であり、優秀な医師アシスタントです。
以下の論文について、忙しい循環器内科医が短時間で把握できるよう、日本語で簡潔な批判的要約を作成してください。

{paper_info}

以下の形式で出力してください。
※「承知いたしました」「要約します」等といった前置きや挨拶は一切含めず、いきなり「## サマリーインデックス情報」から出力してください。

## サマリーインデックス情報
冒頭にインデックスを作成するため、以下の3点を極めて簡潔に出力してください。
- **重要度**: [重要度の基準]に従い、★を5つ並べて表記（例：★★★★★）
- **結論**: [40文字以内]で、この論文が何を示したか
- **実用**: [50文字以内]で、明日の臨床にどう活きるか。疾患名、薬剤名、具体的な数値などの重要キーワードは必ず**太字**にすること

## まず一言で
この論文が何を示したのかを1〜2文で日本語要約。

## 要点
- 研究デザインと対象（1-2行）
- 主な結果（数値を含む、2-3行）
- 臨床的意義（1-2行）
- 主な限界（1-2行）
- 明日からの診療への示唆（1-2行）

重要度の基準：
★★★★★：明日の診療方針に直結する、必ず読むべきパラダイムシフト
★★★★☆：実用性が高く、知っておくべき重要な知見
★★★☆☆：特定の条件下で役立つ、または興味深い知見
★★☆☆☆：参考程度
★☆☆☆☆：現在の業務への直接的な影響は少ない

重要な注意事項:
- 不確かなことは断定しない
- 根拠が弱い場合は弱いと明記
- 誇張表現は避ける
"""


    def generate_selection_reason(self, paper: Paper) -> str:
        """
        論文選出理由を簡潔に生成する

        Args:
            paper: 対象論文

        Returns:
            選出理由テキスト
        """
        reasons = []

        # ジャーナルランク
        journals = self.config.get("journals", {})
        if paper.journal in journals.get("tier1", []):
            reasons.append(f"トップジャーナル（{paper.journal}）掲載")
        elif paper.journal in journals.get("tier2", []):
            reasons.append(f"主要循環器専門誌（{paper.journal}）掲載")

        # 論文タイプ
        high_types = [
            "Randomized Controlled Trial", "Meta-Analysis",
            "Systematic Review", "Practice Guideline"
        ]
        matched_types = [t for t in paper.pub_types if t in high_types]
        if matched_types:
            reasons.append(f"研究デザインが強固（{', '.join(matched_types)}）")

        # 専門領域マッチ
        primary = self.config.get("specialties", {}).get("primary", [])
        title_lower = paper.title.lower()
        matched_areas = [
            s for s in primary if s.lower() in title_lower
        ]
        if matched_areas:
            reasons.append(
                f"最優先領域に関連（{', '.join(matched_areas)}）"
            )

        if reasons:
            return "選出理由: " + "; ".join(reasons)
        return "選出理由: 臨床的重要性が高いと判断"
