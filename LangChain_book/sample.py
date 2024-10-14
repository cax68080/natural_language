import json
import openai

response = openai.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages=[{"role":"user","content":"iPhone8のリリース日を教えて"},]
)

#print(json.dumps(response,indent=2,ensure_ascii=False))
#print(json.dump(list(response), safe=False))

print(list(response))