import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline

model_id = "cyberagent/open-calm-small"

model = AutoModelForCausalLM.from_pretrained(
    model_id
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id
)    
generator = pipeline("text-generation",model=model,tokenizer=tokenizer)

outs = generator("東京は日本の",max_new_tokens=30)
print(outs[0])


#出力
input = tokenizer("東京は日本の",return_tensors="pt")
tokens = model.generate(**input,max_new_tokens=10,do_sample=False)
output = tokenizer.decode(tokens[0],skip_special_tokens=True)
print(output)

#出力
input = tokenizer("東京は日本の",return_tensors="pt")
tokens = model.generate(**input,max_new_tokens=30,eos_token_id=tokenizer.encode("。"),pad_token_id=tokenizer.pad_token_id,do_sample=True,num_return_sequences=5)

for i in range(5):
    output = tokenizer.decode(tokens[i],skip_special_tokens=True)
    print(output)

#出力
#top5 = torch.topk(out.scores[0][0],5)
#for i in range(5):
#    print(i+1,tokenizer.decode(top5.indices[i]),top5.values[i].item())

#出力
#input = tokenizer("今日は天気が良いですね\n" + "そうですね\n" + "どこかへ行きましょうか。",return_tensors="pt")
#tokens = model.generate(**input,max_new_tokens=20,do_sample=False)
#print(tokenizer.decode(tokens[0],skip_special_tokens=True))
