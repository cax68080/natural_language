import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/open-calm-small"
)

tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/open-calm-small"
)


#出力
input = tokenizer("日本の首都はどこですか？",return_tensors="pt")
tokens = model.generate(**input,max_new_tokens=10,do_sample=False)
print(tokenizer.decode(tokens[0],skip_special_tokens=True))

#出力
out = model.generate(**input,max_new_tokens=1,return_dict_in_generate=True,output_scores=True)
print(out.scores[0].shape)

#出力
top5 = torch.topk(out.scores[0][0],5)
for i in range(5):
    print(i+1,tokenizer.decode(top5.indices[i]),top5.values[i].item())

#出力
input = tokenizer("今日は天気が良いですね\n" + "そうですね\n" + "どこかへ行きましょうか。",return_tensors="pt")
tokens = model.generate(**input,max_new_tokens=20,do_sample=False)
print(tokenizer.decode(tokens[0],skip_special_tokens=True))
