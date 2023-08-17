import os
import chainlit as cl
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import pandas as pd
from scipy import spatial
import openai
from datetime import datetime

huggingfacehub_api_token = os.environ['HFH_API_TOKEN']

file_path = "./files/focus_locus.feather"
repo_id = 'tiiuae/falcon-7b-instruct'
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
query_file = "./files/queries.feather"
stats_file = "./files/stats.feather"
api_key = os.environ['OPENAI_API_KEY']


def find_row(df, query_string):
  result = df[df['query'] == query_string]
  if len(result) > 0:
    return result.iloc[0]
  else:
    return None


def record_query(query: str):
  try:
    stats = pd.read_feather(stats_file)
  except:
    stats = pd.DataFrame(data={"query": [], "date": []})

  stats = pd.concat([stats, pd.DataFrame({"query": [query], "date": [datetime.now()]})],
                    ignore_index=True)

  stats.to_feather(stats_file)


def strings_ranked_by_relatedness(
    query: str,
    api_key: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100) -> tuple[list[str], list[float]]:
  """Returns a list of strings and relatednesses, sorted from most related to least."""
  try:
    queries = pd.read_feather(query_file)
  except:
    queries = pd.DataFrame(data={"query": [], "embedding": []})

  query_row = find_row(queries, query)
  if query_row is None:
    query_embedding_response = openai.Embedding.create(
      api_key=api_key,
      model=EMBEDDING_MODEL,
      input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    query_row = {"query": query, "embedding": query_embedding}

    queries = pd.concat([queries, pd.DataFrame([query_row])],
                        ignore_index=True)
    queries.to_feather(query_file)

  strings_and_relatednesses = [(row["quote"],
                                relatedness_fn(query_row['embedding'],
                                               row["embedding"]))
                               for i, row in df.iterrows()]
  strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
  strings, relatednesses = zip(*strings_and_relatednesses)
  return strings[:top_n], relatednesses[:top_n]


if not os.path.exists(file_path):
  raise f"{file_path} not found"

df = pd.read_feather(file_path)
print(df.head())

llm = ChatOpenAI(temperature=0.9, max_tokens=2000, model_name="gpt-4")

prompt = PromptTemplate(
  input_variables=["nearest", "question"],
  template="""
Summarize these previous comments you have made: {nearest} to answer this question: "{question}". Always speak in first person.
""",
)

llm_chain = LLMChain(llm=llm, prompt=prompt)


@cl.on_message
async def main(message):
  msg = cl.Message(content='...')

  await msg.send()
  nearest, nearness = strings_ranked_by_relatedness(message, api_key, df)

  await cl.Message(author="related comments",
                   content=''.join(
                     list(map(lambda s: f'"{s}"\n', nearest[0:15]))),
                   parent_id=msg.id).send()

  print(','.join(list(map(lambda s: f'"{s}"', nearest[0:30]))))

  response = llm_chain.run({
    'nearest':
    ','.join(list(map(lambda s: f'"{s}"', nearest[0:30]))),
    'question':
    message
  })

  msg.content = response

  await msg.update()

  record_query(message)
