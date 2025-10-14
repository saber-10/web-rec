# Building a Closed and Extractive Question and Answer Language model based on provided Context

The goal of this task is to experiment and discover an language model that is suitable for the task of context based extractive question answering. The evaluation should be done on the validation dataset using Metrics like Exact Match(EM) smd F1 score. Calculate the F1 score for the whole validation set by aversging the score for each context,question and answer triplet.

> The NewsQA Dataset and the evaluation metrics along with some other useful links are provided in the references

Different forms of experimentation and architectural model building are encouranged irrelevant of the performance of the model. Novel Ideation or Smart Experimentaion is highly appreciated.

## References :
- [The NewsQA dataset on Hugging face](https://huggingface.co/datasets/lucadiliello/newsqa)
- [What is NewsQA dataset](https://www.microsoft.com/en-us/research/project/newsqa-dataset/)
- [Roberta-FineTuned on SQuaD](https://huggingface.co/deepset/roberta-base-squad2)
- [Fine Tuning on Custom Data](https://huggingface.co/transformers/v3.2.0/custom_datasets.html#question-answering-with-squad-2-0)
- [Evaluation Metrics](https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1)