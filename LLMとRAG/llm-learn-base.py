import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
model_name = "cyberagent/open-calm-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)

input = tokenizer.encode("私は犬は好きだが鳥は嫌い。",return_tensors="pt")
print(input)
a = [tokenizer.decode(input[0][i]) for i in range(len(input[0]))]
print(a)

output = model(input)
print(type(output))

print(output.logits)

print(output.logits.shape)
