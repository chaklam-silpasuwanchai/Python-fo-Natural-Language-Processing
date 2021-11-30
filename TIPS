## Recipe for Success

Data Science is hard! (and fun and very satisfying if you can do it AT THE END)

Great data scientist excels in twofold: science and engineering.

Science -> Why?
- Respect domain knowledge
- Visualize your datasets (know what is X and y)
  - You should be able to visualize some patterns 
- Understand your datasets (where they are from, how they are collected, and how they are potentially contaminated)
- Form hypotheses (and iteratively refine them as you go)
- Form an intuition which architecture would work well
  - CNN1D - good for signal for getting spatial and temporal footprint; basically window-based classification
  - CNN1D + LSTM - if you believe that previous window affects the next window, then it's a good choice; good for signal
  - LSTM only - good for text, because previous text affects the next text; usually bad for signal, because each sample does not mean anything.  If you want to use LSTM for signal, windowed them first through CNN1D or manually.
  - CNN2D - good for any images. For signal (audio, EEG, or any), first convert to some sort of spectrograms.
  - CNN2D + LSTM - good for videos
  - Attention - can be apply on top of CNN or LSTM or only attention; it's about correlations/similarity.  Solves long range dependency.  Should almost always use it to further increase accuracy.
- Master/Ph.D. thesis asks a lot of whys!  Every decision you make is okay that you don't know whether it's right or wrong (because you are still learning), but you all MUST have a clear "why" in your mind (don't do it because Chaky teaches you or it's something new)

Engineering - How?
- First try a simple model (e.g., SVM)
  - Good for understanding your dataset 
- Then try a simple neural network
  - Good for debugging
- Try add more layers;  ++**slowly**++ and ++**incrementally**++ refine your neural network 
- Monitor your progress
  - Loss plateau (if it is very steep or unstable, use these techniques)
    - learning rate, batch norm, momentum, Adam (adaptive learning algorithm)
  - Gradient variances (if your gradients are very small, or very big)
    - activation, weight initialization, skip connections, batch norm
  - Weights (if your weights are not uniform, or very similar)
    - weight initialization 
- Always MAKE SURE to overfit your training data
- Last thing to complete your model is to change your hyperparameters (quality of life perk)
- To counter overfitting, use regularization L1/L2, dropout, label smoothing, batch norm, skip connections     
- Keeping your results in tabular format (e.g., excel) is a really good way for communication
- How to train-test split is EXTREMELY important; but regardless of whatever way you do, please make sure you pass these different levels of tests (i.e., increasing difficulty): generalization across (1) within him/herself, (2) participants, (3) tasks, and (4) domain.
- Always reserve a testing set that is NEVER seen by the model
- Look for inspirations
  - Read papers; read 25 papers for good results, read 100 papers for breakthroughs
  - Look at github; many nice guys out there
- Deep learning is still a craft (not entirely 100% science); researchers are still understanding it, so it is very normal that the formulation of the architecture is very much a trial and error process, but there is always something call "intuition" or "gut feeling that it will work" behind - read papers to learn this intuition.  TA and myself don't always have answers to your questions because there is no definite science one-for-all solution behind most of these deep learning stuffs.
