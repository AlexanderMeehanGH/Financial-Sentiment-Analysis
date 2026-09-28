FINANCIAL NEWS SENTIMENT ANALYSIS — COMPARING BERT, FINBERT & GENERATIVE AI
[University], BAA1053 Advanced Analytics

THE PROBLEM
Financial markets react to news, and the modern 24-hour news cycle produces far more headlines than analysts can read manually. Investment firms need a reliable, automated way to decide whether a headline signals positive, neutral or negative sentiment toward a company or market. The key question was which kind of model does this best: a general-purpose language model, a finance-specific model, or a modern generative AI model.

MY APPROACH
I worked with a dataset of about 4,800 financial news sentences labelled for sentiment. From there, I:
- Explored and cleaned the data. I found a heavy class imbalance (59% neutral), which meant a model could look accurate by always guessing "neutral". I corrected for this with a balanced train/test split and class-weighted loss.
- Fine-tuned two transformer models, BERT-base and FinBERT, using identical hyperparameters. This way, the only difference between them was their pre-training data (general text vs financial text).
- Tested Google's Gemma 3, run locally through Ollama, as a zero-shot classifier with no training, to see how generative AI compares with fine-tuned models.
- Evaluated all three on the same test set using accuracy, macro F1 and confusion matrices.

WHAT I BUILT
I built an end-to-end Python pipeline in PyTorch and Hugging Face covering data cleaning, tokenisation, model training, LLM prompting and evaluation. It also produces visual outputs: training curves, performance comparisons and confusion matrices. These make it easy to see where each model succeeds and where it misclassifies headlines.

THE OUTCOME
- FinBERT performed best (88.4% accuracy, 0.88 F1), ahead of BERT-base (84.7%) and Gemma 3 (81.4%). This shows the value of domain-specific pre-training.
- Gemma 3 was the strongest at catching negative headlines (177 of 181), which is useful for risk monitoring, but it struggled to recognise neutral news.
- I recommended FinBERT for deployment within a human-in-command framework, where the model flags sentiment and analysts make the decisions, given its roughly 1-in-9 error rate. I also identified few-shot prompting of Gemma 3 as a next step.
