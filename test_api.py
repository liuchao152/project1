from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
with open("D:\\bishe\\CHIP\\2.txt", "r", encoding="utf-8") as f:
    file_content = f.read()
try:
    resp = client.chat.completions.create(
        model="qwen3-32b",
        messages=[{"role": "user", "content": file_content}],
        temperature=0.7,
        max_tokens=10000,
        extra_body={
            "enable_thinking": False
        }
    )
    print("✅ API 调用成功！")
    print(f"回复：{resp.choices[0].message.content}")
except Exception as e:
    print(f"❌ API 调用失败：{e}")