import dspy
from dspy.teleprompt import MIPROv2
import os
import json

# --- 1. Dataset Preparation ---
# 数学の問題と解答のペア
math_data = [
    dspy.Example(question="What is 10 + 20?", answer="30").with_inputs("question"),
    dspy.Example(question="What is 5 * 5?", answer="25").with_inputs("question"),
    dspy.Example(question="What is 100 / 2?", answer="50").with_inputs("question"),
    dspy.Example(question="What is 12 - 4?", answer="8").with_inputs("question"),
    dspy.Example(question="What is 3 + 2 * 4?", answer="11").with_inputs("question"),
]
trainset = math_data[:3] # 訓練用
valset = math_data[3:]   # 検証用

# --- 2. Module Definition ---
class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought: ステップバイステップの推論を行う
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

# --- 3. Metric Definition ---
# 正解判定ロジック
def validate_math(example, pred, trace=None):
    return example.answer in pred.answer

def main():
    # APIキーの確認
    if not os.environ.get("GOOGLE_API_KEY"):
        print("エラー: GOOGLE_API_KEY環境変数が設定されていません。")
        print("実行前に設定してください: export GOOGLE_API_KEY='AIza...'")
        return

    # Language Modelの設定 (Gemma 3 27b)
    lm = dspy.LM('gemini/gemma-3-27b-it')
    dspy.configure(lm=lm)

    print("\n--- Initializing MIPROv2 (Prompt + Few-Shot Optimization) ---")
    
    # MIPROv2の初期化
    # num_candidates: 生成するプロンプト候補の数 (デモ用に少なく設定)
    # auto: Noneに設定することでnum_candidatesを自前で指定可能にする
    teleprompter = MIPROv2(metric=validate_math, num_candidates=3, auto=None)
    
    # コンパイル（学習）の実行
    # max_bootstrapped_demos=2: 自分で生成した成功例（ブートストラップデモ）を最大2つ使用
    # max_labeled_demos=2: トレーニングデータからの正解例（ラベル付きデモ）を最大2つ使用
    # これにより、MIPROv2は最適なインストラクションと最適な事例の組み合わせを探索します
    print("Compiling with MIPROv2 (Instructions + Few-Shot)...")
    compiled_program = teleprompter.compile(
        student=MathSolver(),
        trainset=trainset,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        minibatch=False,
        num_trials=6,
    )

    # --- Optimized Execution ---
    print("\n--- Optimized Execution ---")
    question = "What is 15 + 5 * 2?"
    pred_optimized = compiled_program(question)
    print(f"Question: {question}")
    print(f"Answer: {pred_optimized.answer}")
    print(f"Reasoning: {pred_optimized.reasoning}")

    # Capture optimized prompt
    optimized_prompt = "No history"
    if lm.history:
        last_entry = lm.history[-1]
        if "messages" in last_entry:
            optimized_prompt = last_entry["messages"]
        elif "prompt" in last_entry:
            optimized_prompt = last_entry["prompt"]
        else:
            optimized_prompt = str(last_entry)

    print("\n" + "="*20 + " Optimized Prompt (Instruction + FewShot) " + "="*20)
    print(json.dumps(optimized_prompt, indent=2, ensure_ascii=False))
    print("="*78)

    # プログラムの保存（オプション）
    # compiled_program.save("math_solver_mipro_full.json")

if __name__ == "__main__":
    main()
