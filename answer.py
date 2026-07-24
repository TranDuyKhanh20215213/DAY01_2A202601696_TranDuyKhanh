from template import call_openai

prompt = "Hãy kể cho tôi một sự thật thú vị về Hà Nội."
temperatures = [0.0, 0.7, 1.2, 1.8]

def main():
    print(f"Prompt: '{prompt}'\n")
    for temp in temperatures:
        print("=" * 60)
        print(f"Temperature: {temp}")
        print("=" * 60)
        try:
            response_text, latency = call_openai(prompt, temperature=temp, max_tokens=2048)
            print(f"Độ trễ (latency): {latency:.2f}s")
            print(f"Phản hồi:\n{response_text}\n")
        except Exception as e:
            print(f"Lỗi: {e}\n")

if __name__ == "__main__":
    main()
