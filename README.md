# Question-Answering-BERT


BERT is a relatively recent innovation in deep learning field, but it has already inspired a lot of other models (such as Transformer XL by Google, GPT-2 by OpenAI and XLNet). The main motivation I have for using this network is its ability to classify intention. As shown in multiple publications, classical long-short term memory networks (LSTMs) are not as good with this tasks, mainly because they lack the concept of attention.

The repo contains the model pretrained on the SQuAD (https://rajpurkar.github.io/SQuAD-explorer/) that is then used to answer  factual questions. The reference for answers is taken from Wikipedia articles. Further improvement is needed to pick better references. 


Examples of how it works:

![Examples 1](/pictures/ex1.png)

![Example 2](/pictures/ex2.png)
