## Diagnosing Optimization During Fine-Tuning

### Project Description
This project fine-tunes DistilBERT using AdamW and Muon on the SST-2 sentiment classification dataset. Since the original BERT paper was trained on the SST-2 dataset, DistilBERT was used to fulfill the requirement of a smaller model.

### Optimization and Performance Metrics
There were quite a few metrics that were recorded for specific reasons. 

1. **periodic_test_eval.png** *This metric was recorded to get the test evals in between epochs. Since fine-tuning doesn't require as many epochs, and the models tend to overfit after few epochs, recording at every n steps allows to see how the model is performing on the test set in between epochs. The main metric to tune the frequency of logging is* `eval_freq` *in* `train.py`. *The current value is set to* `100`.
2. **rotational_equilibrium.png** *This metric, according to* https://arxiv.org/pdf/2305.17212, *shows the true learning rate. It's important for measuring behaviors between optimizers.*
3. **loss_steps.png** *This metric is a basic optimization trajectory to check for stability and behaviors.*
4. **distance_pretrain_weights.png** *This metric answers of how post-training is affecting the pre-trained weights. This metric was also suggested by https://arxiv.org/pdf/2605.10468*.

`plots.py` does generate 4 more graphs, but they are