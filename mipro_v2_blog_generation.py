import dspy
from dspy.teleprompt import MIPROv2
import os
import glob
import json

# --- 1. Dataset Loading ---
def load_dataset():
    """datasetフォルダからノートとブログのペアを読み込む"""
    notes_path = "dataset/notes/*.txt"
    dataset = []
    
    for note_file in glob.glob(notes_path):
        # ファイル名からベース名を取得 (例: "remote_work")
        base_name = os.path.basename(note_file)
        blog_file = os.path.join("dataset/blogs", base_name)
        
        if os.path.exists(blog_file):
            with open(note_file, 'r', encoding='utf-8') as f:
                notes_text = f.read().strip()
            with open(blog_file, 'r', encoding='utf-8') as f:
                blog_text = f.read().strip()
            
            # Exampleを作成。notesを入力、blogを正解データとする
            example = dspy.Example(notes=notes_text, blog=blog_text).with_inputs("notes")
            dataset.append(example)
    
    return dataset

# データを読み込み
full_dataset = load_dataset()
if not full_dataset:
    print("Error: データが見つかりません。dataset/notes と dataset/blogs にファイルがあるか確認してください。")
    exit()

# 訓練用データとして全て使用（デモのため）
trainset = full_dataset

# --- 2. Module Definition (The Student) ---
class BlogWriter(dspy.Module):
    """
    あなたはプロのSEOライターです。以下のテーマに基づいて、読者の検索意図を満たすブログ記事を執筆してください。
    
    【制約事項】
    - 文体は「です・ます」調で、親しみやすく書いてください。
    - 結論から書き始め（PREP法）、具体例を交えてください。
    - Markdown形式で見出しを適切に使ってください。
    - 専門用語には簡単な解説を入れてください。
    """
    def __init__(self):
        super().__init__()
        # 思考の連鎖(CoT)を使ってブログを書く
        # 入力: notes (箇条書きメモ)
        # 出力: blog (ブログ記事)
        self.prog = dspy.ChainOfThought("notes -> blog", desc="Write a blog post in Japanese based on the notes.")
    
    def forward(self, notes):
        return self.prog(notes=notes)

# --- 3. Metric Definition (LLM-as-a-Judge) ---
# 評価を行うための署名（Signature）
class BlogJudgeSignature(dspy.Signature):
    """
    あなたは専門の編集者です。
    提供されたメモに基づいて、生成されたブログ記事を評価してください。

    評価基準:
    1. 言語: **必ず日本語で書かれていること**。英語が含まれていても良いが、メインは日本語であること。
    2. 網羅性: メモにある主要なポイントがすべてカバーされているか？
    3. 品質: 文章が魅力的で、専門的で、よく構成されているか？
    4. 形式: ブログ記事としての形式（見出しなど）が整っているか？
    
    1から5までのスコアを付けてください（日本語でない場合は1にしてください）。
    """
    notes = dspy.InputField(desc="執筆者に提供された元のメモ")
    generated_blog = dspy.InputField(desc="執筆者によって生成されたブログ記事")
    reasoning = dspy.OutputField(desc="スコアの理由")
    score = dspy.OutputField(desc="1から5までの整数スコア", dtype=int)

# 評価関数
def validate_blog(example, pred, trace=None):
    # 簡易チェック: ひらがなが含まれていなければ日本語ではないとみなして即不合格（False）
    # これにより、英語で出力されたものを確実に弾く
    import re
    if not re.search(r'[ぁ-ん]', pred.blog):
        return False

    # 評価用の小さなモジュールを用意
    judge = dspy.ChainOfThought(BlogJudgeSignature)
    
    try:
        result = judge(notes=example.notes, generated_blog=pred.blog)
        score = int(result.score)
        
        # スコアそのものを返すことで、より高いスコアを目指させる（MIPROv2は数値を最大化しようとする）
        return score
    except Exception as e:
        print(f"Metric Error: {e}")
        return 0

def main():
    # APIキーの確認
    if not os.environ.get("GOOGLE_API_KEY"):
        print("エラー: GOOGLE_API_KEY環境変数が設定されていません。")
        return

    # Language Modelの設定
    lm = dspy.LM('gemini/gemma-3-27b-it')
    dspy.configure(lm=lm)

    print(f"\n--- Loaded {len(trainset)} examples from dataset/ folder ---")

    # --- Capture Initial Prompt ---
    print("\n--- Capturing Initial Prompt (Before Optimization) ---")
    test_notes = """
    - 良い睡眠をとるためには、寝る前のスマホを控える
    - お風呂は就寝の90分前に済ませるのが理想
    - 室温は20度〜25度くらいが快適
    - 朝起きたら日光を浴びて体内時計をリセットする
    """
    
    # 最適化前のモジュールで一度実行
    initial_writer = BlogWriter()
    initial_writer(test_notes)
    
    initial_prompt = "No history"
    if lm.history:
        last_entry = lm.history[-1]
        initial_prompt = last_entry.get("messages", last_entry.get("prompt", str(last_entry)))

    print("\n--- Initializing MIPROv2 (Zero-Shot Prompt Optimization) ---")
    
    # MIPROv2の初期化
    # MetricにLLM審査員を使用
    teleprompter = MIPROv2(metric=validate_blog, num_candidates=3, auto=None)
    
    print("Compiling with MIPROv2...")
    print("Objective: Find the best Zero-shot instruction to turn notes into a blog.")
    
    # コンパイル（学習）
    # Zero-shot最適化のため、デモ数は0に設定
    compiled_blog_writer = teleprompter.compile(
        student=BlogWriter(),
        trainset=trainset,
        max_bootstrapped_demos=0, # 事例生成なし
        max_labeled_demos=0,      # 教師データ事例の使用なし
        minibatch=False,
        num_trials=4, # デモ用に試行回数を減らす
    )

    # --- Optimized Execution ---
    print("\n--- Optimized Execution Check ---")
    print(f"Input Notes:\n{test_notes}")
    pred = compiled_blog_writer(test_notes)
    
    print("\n" + "="*20 + " Generated Blog " + "="*20)
    print(pred.blog)
    print("="*56)
    
    # Capture optimized prompt
    optimized_prompt = "No history"
    if lm.history:
        last_entry = lm.history[-1]
        optimized_prompt = last_entry.get("messages", last_entry.get("prompt", str(last_entry)))

    print("\n" + "="*20 + " Initial Prompt (Before) " + "="*20)
    print(json.dumps(initial_prompt, indent=2, ensure_ascii=False))
    
    print("\n" + "="*20 + " Optimized Prompt (After) " + "="*20)
    print(json.dumps(optimized_prompt, indent=2, ensure_ascii=False))
    print("="*70)

    # 保存
    # compiled_blog_writer.save("blog_writer_optimized.json")

if __name__ == "__main__":
    main()
