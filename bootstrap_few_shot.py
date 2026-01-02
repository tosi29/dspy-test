import dspy
from dspy.teleprompt import BootstrapFewShot
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
    # 簡単のため、文字列の完全一致または数値が含まれているかで判定も可能だが
    # ここでは単純に正解文字列が予測に含まれているかチェック
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

    # --- 4. Unoptimized Execution (Zero-shot) ---
    print("\n--- Zero-shot Execution ---")
    unoptimized_program = MathSolver()
    question = "What is 15 + 5 * 2?"
    pred = unoptimized_program(question)
    print(f"Question: {question}")
    print(f"Answer: {pred.answer}")
    # Capture unoptimized prompt
    # print(f"DEBUG: lm.history[-1] keys: {lm.history[-1].keys() if lm.history else 'History is empty'}")
    # print(f"DEBUG: lm.history[-1]: {lm.history[-1] if lm.history else 'History is empty'}")
    
    unoptimized_prompt = "No history"
    if lm.history:
        last_entry = lm.history[-1]
        if "messages" in last_entry:
            unoptimized_prompt = last_entry["messages"]
        elif "prompt" in last_entry:
            unoptimized_prompt = last_entry["prompt"]
        else:
            unoptimized_prompt = str(last_entry) # Fallback

    # --- 5. Optimization (BootstrapFewShot) ---
    print("\n--- Optimizing... ---")
    # BootstrapFewShot: 教師データを使ってFew-shotの例を自分自身で生成・選定する
    teleprompter = BootstrapFewShot(metric=validate_math, max_bootstrapped_demos=2)
    
    # コンパイル（学習）の実行
    compiled_program = teleprompter.compile(student=MathSolver(), trainset=trainset)

    # --- 6. Optimized Execution ---
    print("\n--- Optimized Execution ---")
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

    # 生成されたデモ（Few-shot例）を検査したい場合
    print("\n--- Prompt History ---")
    # dspy.inspect_history(n=1)

    print("\n" + "="*20 + " Pre-Optimization Prompt " + "="*20)
    print(json.dumps(unoptimized_prompt, indent=2, ensure_ascii=False))
    print("="*65)

    print("\n" + "="*20 + " Post-Optimization Prompt " + "="*20)
    print(json.dumps(optimized_prompt, indent=2, ensure_ascii=False))
    print("="*66)

if __name__ == "__main__":
    main()
