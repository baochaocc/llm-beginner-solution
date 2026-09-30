---
license: apache-2.0
---

<p align="right">
  <a href="https://github.com/argilla-io/distilabel">
    <img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>
  </a>
</p>

# distilabel Orca Pairs for DPO

The dataset is a "distilabeled" version of the widely used dataset: [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs). The original dataset has been used by 100s of open-source practitioners and models. We knew from fixing UltraFeedback (and before that, Alpacas and Dollys) that this dataset could be highly improved.

Continuing with our mission to build the best alignment datasets for open-source LLMs and the community, we spent a few hours improving it with [distilabel](https://github.com/argilla-io/distilabel). 

This was our main intuition: the original dataset just assumes gpt4/3.5-turbo are always the best response. We know from UltraFeedback that's not always the case. Moreover, DPO fine-tuning benefits from the diversity of preference pairs. 

Additionally, we have added a new column indicating whether the question in the dataset is part of the train set of gsm8k (there were no examples from the test set). See the reproduction section for more details.

## Using this dataset
This dataset is useful for preference tuning and we recommend using it instead of the original. It's already prepared in the "standard" chosen, rejected format with additional information for further filtering and experimentation. 

The main changes are:

1. ~2K pairs have been swapped: rejected become the chosen response. We have kept the original chosen and rejected on two new columns `original_*` for reproducibility purposes.
2. 4K pairs have been identified as `tie`: equally bad or good.
3. Chosen scores have been added: you can now filter out based on a threshold (see our distilabeled Hermes 2.5 model for an example)
4. We have kept the ratings and rationales generated with gpt-4-turbo and distilabel so you can prepare the data differently if you want.
5. We have added a column to indicate if the input is part of gsm8k train set.

In our experiments, we have got very good results by reducing the size of the dataset by more than 50%. Here's an example of how to achieve that:

```python
from datasets import load_dataset

# Instead of this:
# dataset = load_dataset("Intel/orca_dpo_pairs", split="train")

# use this:
dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")

dataset = dataset.filter(
    lambda r: 
        r["status"] != "tie" and 
        r["chosen_score"] >= 8 and 
        not r["in_gsm8k_train"]
)
```
This results in `5,922` instead of `12,859` samples (54% reduction) and leads to better performance than the same model tuned with 100% of the samples in the original dataset.

> We'd love to hear about your experiments! If you want to try this out, consider joining our [Slack community](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g) and let's build some open datasets and models together.


## Reproducing the dataset
In this section, we outline the steps to reproduce this dataset.

### Rate original dataset pairs

Build a preference dataset with distilabel using the original dataset:

```python
from distilabel.llm import OpenAILLM
from distilabel.tasks import JudgeLMTask
from distilabel.pipeline import Pipeline

from datasets import load_dataset

# Shuffle 'chosen' and 'rejected' to avoid positional bias and keep track of the order
def shuffle_and_track(chosen, rejected):
    pair = [chosen, rejected]
    random.shuffle(pair)
    order = ["chosen" if x == chosen else "rejected" for x in pair]
    return {"generations": pair, "order": order}

dataset = load_dataset("Intel/orca_dpo_pairs", split="train")

# This shuffles the pairs to mitigate positional bias
dataset = dataset.map(lambda x: shuffle_and_track(x["chosen"], x["rejected"]))

# We use our JudgeLM implementation to rate the original pairs
labeler = OpenAILLM(
    task=JudgeLMTask(),
    model="gpt-4-1106-preview",
    num_threads=16,
    max_new_tokens=512,
)

dataset = dataset.rename_columns({"question": "input"})

distipipe = Pipeline(
    labeller=labeler
)

# This computes ratings and natural language critiques for each pair
ds = distipipe.generate(dataset=dataset, num_generations=2)
```

If you want to further filter and curate the dataset, you can push the dataset to [Argilla](https://github.com/argilla-io/argilla) as follows:

```python
rg_dataset = ds.to_argilla()

rg_dataset.push_to_argilla(name="your_dataset_name", workspace="your_workspace_name")

```
You get a nice UI with a lot of pre-computed metadata to explore and curate the dataset:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/IoK4nFObadhJpkVmWALZP.png)

The resulting dataset is now much more useful: we know which response is preferred (by gpt-4-turbo), which ones have low scores, and we even have natural language explanations. But what did we find? Was our intuition confirmed?

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/-V8wY1DYzrtwM9LbGrBXq.png)

The above chart shows the following: 

* ~4,000 pairs were given the same rating (a tie).
* ~7,000 pairs were correct according to our AI judge (`unchanged`).
* and ~2,000 times the rejected response was preferred (`swapped`).



Now the next question is: can we build better models with this new knowledge? The answer is the "distilabeled Hermes" model, check it out!


### Post-processing to add useful information


Swap rejected and chosen, and add chosen scores and status:

```python
def add_status(r):
  status = "unchanged"
  highest_rated_idx = np.argmax(r['rating'])
  # Compare to the index of the chosen response
  if r['rating']== None or r['rating'][0] == r['rating'][1]:
    status = "tie"
  elif r['order'][highest_rated_idx] != 'chosen':
      status = "swapped"
  return {"status": status}

def swap(r):
  chosen = r["chosen"]
  rejected = r["rejected"]
  if r['rating'] is not None:
    chosen_score = r['rating'][np.argmax(r['rating'])]
  else:
    chosen_score = None
  if r['status'] == "swapped":
    chosen = r["rejected"]
    rejected = r["chosen"]
  return {
      "chosen": chosen,
      "rejected": rejected,
      "original_chosen": r["chosen"],
      "original_rejected": r["rejected"],
      "chosen_score": chosen_score
  }

updated = ds.map(add_status).map(swap)
```

### gsm8k "decontamination"
The basic approach for finding duplicated examples. We didn't find any from the test sets. We experimented with lower thresholds but below 0.8 they introduced false positives:
```python
import pandas as pd

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import load_dataset

nltk.download('punkt')

# Load the datasets
source_dataset = load_dataset("gsm8k", "main", split="train")
source_dataset_socratic = load_dataset("gsm8k", "socratic", split="train")
#target_dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
target_dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")

# Extract the 'question' column from each dataset
source_questions = source_dataset['question']
source_questions_socratic = source_dataset_socratic['question']
target_questions = target_dataset['input']

# Function to preprocess the text
def preprocess(text):
    return nltk.word_tokenize(text.lower())

# Preprocess the questions
source_questions_processed = [preprocess(q) for q in source_questions]
source_questions.extend([preprocess(q) for q in source_questions_socratic])
target_questions_processed = [preprocess(q) for q in target_questions]


# Vectorize the questions
vectorizer = TfidfVectorizer()
source_vec = vectorizer.fit_transform([' '.join(q) for q in source_questions_processed])
target_vec = vectorizer.transform([' '.join(q) for q in target_questions_processed])

# Calculate cosine similarity
similarity_matrix = cosine_similarity(source_vec, target_vec)

# Determine matches based on a threshold:
# checked manually and below 0.8 there are only false positives
threshold = 0.8
matching_pairs = []
for i, row in enumerate(similarity_matrix):
    for j, similarity in enumerate(row):
        if similarity >= threshold:
            matching_pairs.append((source_questions[i], target_questions[j], similarity))

# Create a DataFrame from the matching pairs
df = pd.DataFrame(matching_pairs, columns=['Source Question', 'Target Question', 'Similarity Score'])

# Create a set of matching target questions
matching_target_questions = list(df['Target Question'])

# Add a column to the target dataset indicating whether each question is matched
target_dataset = target_dataset.map(lambda example: {"in_gsm8k_train": example['input'] in matching_target_questions})
```
Result:
```
False    12780
True        79
Name: in_gsm8k_train
```