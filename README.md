## Diagnosing Optimization During Fine-Tuning

### DISCLAIMER
All but `plot_sharpness.py` is hand-coded. This is due to the fact that the author is not as familiar with `matplotlib`. Even this `README.md` is hand-written. However, `Claude Fable 5.1 Medium` was used as a verbal assistant to guide the author to plan, find papers, and to understand the logic and math behind some of the functions. In addition, it was used to code `plot_sharpness.py`. **FOR SETUP INSTRUCTIONS, SCROLL TO THE BOTTOM.**

### Project Description
This project fine-tunes DistilBERT using AdamW and Muon on the SST-2 sentiment classification dataset. Since the original BERT paper was trained on the SST-2 dataset, DistilBERT was used to fulfill the requirement of a smaller model.

### Optimization and Performance Metrics
There were quite a few metrics that were recorded for specific reasons. 

1. **periodic_test_eval.png** This metric was recorded to get the test evals in between epochs. Since fine-tuning doesn't require as many epochs, and the models tend to overfit after few epochs, recording at every n steps allows to see how the model is performing on the test set in between epochs. The main metric to tune the frequency of logging is `eval_freq` in `train.py`. The current value is set to `100`.
2. **rotational_equilibrium.png** This metric, according to https://arxiv.org/pdf/2305.17212, shows the true learning rate. It's important for measuring behaviors between optimizers.
3. **loss_steps.png** This metric is a basic optimization trajectory to check for stability and behaviors.
4. **distance_pretrain_weights.png** This metric answers of how post-training is affecting the pre-trained weights. This metric was also suggested by https://arxiv.org/pdf/2605.10468.

`plots.py` does generate 3 more graphs: **gradient_norms.png**, **update_norms.png**, and **weight_norms.png**, each used for input, output and denominators for the **rotational_equilibrium.png**. It also generates a summary table with several metrics to highlight the differences between the two optimizers used (AdamW and Muon).

### Sharpness Metrics
For sharpness, 3 different methods were used.

1. **perturbation_sharpness.png** This metric was inspired by* https://arxiv.org/pdf/1609.04836. This metric looks around the two different post-trained model weights in the weight space to see the steepness of the landscape around where the weights ended up. It's like throwing a ball and seeing whether the hole that it fell into was wide or thin. The `sigmas` and `n_draws` metrics in `sharpness.py` affect the nudge distance and accuracy of std.
2. **interpolation.png** This is a very important graph for this experiment, as it answers the main question as to how reliable the results from the Optimization and Performance Metrics is. This is a standard metric to view the landscape between the two models' weights. Currently, `"ts": np.linspace(-0.5, 1.5, 25)` in `sharpness.py` for this metric, which means that if you were to picture a number line, where the AdamW model is at 0 and the Muon model is at 1, then this metric is covering the landscape in a line from point -0.5 to 1.5, covering an extra 50% on both ends to see the landscape before the AdamW model and after the Muon model.
3. `top_eigenvalue()` in `sharpness.py` is the last method of evaluation. What it means is to take the value of the steepest descent, climbing the tallest "wall" in the basin of the model it is trying to escape. `"n_iters": 20` is the metric that matters here, which means that it tried to backpropagate 20 times to reach the highest loss. 

There was no graph for the `top_eigenvalue()` method. Instead there was a table. 

### Results
