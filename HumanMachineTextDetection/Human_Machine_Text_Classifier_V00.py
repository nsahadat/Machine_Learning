
import pandas as pd
import numpy as np
import os
import sys
from transformers import AutoTokenizer

# from datasets import load_dataset

# test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# import torch
# from tqdm import tqdm

# max_length = model.config.n_positions
# stride = 256
# seq_len = encodings.input_ids.size(1)

# nlls = []
# prev_end_loc = 0
# for begin_loc in tqdm(range(0, seq_len, stride)):
#     end_loc = min(begin_loc + max_length, seq_len)
#     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#     input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#     target_ids = input_ids.clone()
#     target_ids[:, :-trg_len] = -100

#     with torch.no_grad():
#         outputs = model(input_ids, labels=target_ids)

#         # loss is calculated using CrossEntropyLoss which averages over valid labels
#         # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
#         # to the left by 1.
#         neg_log_likelihood = outputs.loss

#     nlls.append(neg_log_likelihood)

#     prev_end_loc = end_loc
#     if end_loc == seq_len:
#         break

# ppl = torch.exp(torch.stack(nlls).mean())

# Use a pipeline as a high-level helper
from transformers import pipeline


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

from datasets import load_dataset, concatenate_datasets
# 'wiki_labeled', 'research_abstracts_labeled'
dataset1 = load_dataset("NicolaiSivesind/human-vs-machine", "wiki_labeled")
dataset2 = load_dataset("NicolaiSivesind/human-vs-machine", "research_abstracts_labeled")


dataset_train = concatenate_datasets([dataset1['train'], dataset2['train']])
dataset_validation = concatenate_datasets([dataset1['validation'], dataset2['validation']])

print(dataset_train)
print(dataset_validation)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_data_train = dataset_train.map(tokenize_function, batched=True)
tokenized_data_validation = dataset_validation.map(tokenize_function, batched=True)

tokenized_data_train

tokenized_data_validation

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, label = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=label)


from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir="test_trainer",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data_train, #.shuffle(seed=42).select(range(1000)),
    eval_dataset=tokenized_data_validation, #.shuffle(seed=42).select(range(1000)),
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


trainer.train()
trainer.save_model('./test_trainer/human_machine_classifier')
print(trainer.state.best_model_checkpoint)


from transformers import pipeline,TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

pipe = pipeline('text-classification', model='./test_trainer/human_machine_classifier')
# prediction = pipe("The text to predict", return_all_scores=True)

test = ["""Collaborative search on the plane without communication presents a novel approach to solving the task of finding a target on a flat surface, without the use of communication among the searchers. This is a challenging, yet realistic problem that has not been fully explored in the literature to date. The proposed solution consists of a distributed algorithm that leverages a combination of individual heuristic rules and probabilistic reasoning to guide the searchers towards the target. Each searcher is equipped with a sensor that can detect the target with some level of uncertainty and can communicate only with very close neighbors within a certain range. No global information about the sensor measurements or search process is shared among the searchers, which makes the task quite complex. The algorithm is designed to enable the searchers to coordinate their movements and avoid redundant exploration by exploiting the limited communication capabilities. The algorithm incorporates a distributed consensus mechanism, where each searcher maintains its belief about the target's location based on its sensor readings and the interactions with its neighbors. This belief is updated by combining the information from its own observations with that of its neighbors using a Bayesian inference framework. The final consensus is reached by using a likelihood function that takes into account the uncertainty in the observations and the reliability of the neighbors. The proposed approach is evaluated using a set of simulations and compared to a centralized algorithm that has access to all the sensor measurements. The results show that the proposed algorithm is able to achieve comparable performance to the centralized algorithm, while using only local information and limited communication. Moreover, the proposed algorithm is shown to be scalable and robust to changes in the search environment, such as the disappearance and sudden reappearance of the target. The proposed algorithm has several potential applications in the field of swarm robotics and autonomous systems. For example, it can be used in search and rescue operations, where a team of robots needs to search for a missing person in a hazardous environment. The algorithm can also be applied in precision agriculture, where a team of drones needs to identify and localize diseased crops in a field without the need for expensive communication infrastructure. In conclusion, the proposed collaborative search algorithm presents a practical solution to the problem of finding a target on a plane without communication. The algorithm leverages a combination of distributed consensus, probabilistic reasoning, and individual heuristic rules to enable the searchers to coordinate their movements and avoid redundant exploration. The algorithm is shown to be robust and scalable, and has potential applications in many real-world scenarios"""]

pipe(test, return_all_scores = True)[0][1]['score']
