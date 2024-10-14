from langchain_core.prompts import (PromptTemplate,)

prompte_template = PromptTemplate.from_template("与えた単語を{language}に変換してください。")
result = prompte_template.invoke({"language":"日本語"})

print(result)