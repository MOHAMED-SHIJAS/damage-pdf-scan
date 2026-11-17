"""
Vision & Multimodal Workflow Automation Agent
==============================================
Main Root Entry Point: main.py
Built in Plain Python (No LangChain, CrewAI, or heavy agent frameworks).

Usage:
    python main.py 1   # Run Part 1: The Core ReAct Loop
    python main.py 2   # Run Part 2: 2-4 Tools Registry & Execution
    python main.py 3   # Run Part 3: Self-Healing & Error Handling
    python main.py     # Interactive Menu
"""

import sys
import os

# Ensure the agent package is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def banner():
    print("=" * 70)
    print("🤖 VISION & MULTIMODAL WORKFLOW AUTOMATION AGENT")
    print("   Pure Python Implementation · Zero External Agent Frameworks")
    print("=" * 70)


def show_menu():
    banner()
    print("\nSelect a part to execute:")
    print("  [1] Part 1: The Core Loop (Think -> Act -> Look)")
    print("  [2] Part 2: 2-4 Tools Registry & Execution")
    print("  [3] Part 3: Fixing Errors When a Tool Fails (Self-Healing)")
    print("  [4] Part 4: Memory Between Tasks (Persistent memory_store.json)")
    print("  [5] Part 5: A Plan Before It Acts (Explicit Planning Engine)")
    print("  [6] Part 6: Safety (Ask First, Step Limit, Audit Log)")
    print("  [7] Part 7: Two-Tier Tests (Fake LLM -> Real LLM)")
    print("  [8] Part 8 (⭐ Bonus): Team of Agents (Multi-Agent Collaboration)")
    print("  [w] Launch Web UI (Streamlit Dashboard: app.py)")
    print("  [t] Inspect Registered Tools Catalog")
    print("  [q] Quit")
    print("-" * 70)


def run_part(part_num: str):
    banner()
    if part_num == "1":
        from agent.run_part1 import main as run_p1
        run_p1()
    elif part_num == "2":
        from agent.run_part2 import main as run_p2
        run_p2()
    elif part_num == "3":
        from agent.run_part3 import main as run_p3
        run_p3()
    elif part_num == "4":
        from agent.run_part4 import main as run_p4
        run_p4()
    elif part_num == "5":
        from agent.run_part5 import main as run_p5
        run_p5()
    elif part_num == "6":
        from agent.run_part6 import main as run_p6
        run_p6()
    elif part_num == "7":
        from agent.run_part7 import main as run_p7
        run_p7()
    elif part_num in ("8", "bonus"):
        from agent.run_part8 import main as run_p8
        run_p8()
    elif part_num.lower() in ("w", "web", "ui", "streamlit", "app"):
        import subprocess
        print("\n🚀 Launching Streamlit Web App (app.py)...")
        print("   If it does not open automatically, visit http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    elif part_num.lower() in ("t", "tools"):
        from agent.tools import default_registry
        print("\n📦 REGISTERED TOOLS CATALOG:")
        print(default_registry.get_descriptions())
    else:
        print(f"Unknown part '{part_num}'. Choose 1 to 8, or t.")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].replace("--part", "").replace("-p", "").strip()
        run_part(arg)
    else:
        show_menu()
        choice = input("Enter your choice (1-8, w, t, q): ").strip()
        if choice.lower() != "q":
            run_part(choice)


if __name__ == "__main__":
    main()

