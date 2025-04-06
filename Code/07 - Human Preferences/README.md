# Reinforcement Learning with Human Feedback

1. **Custom Dataset:** This model, which involves the creation and utilization of a custom dataset, can be considered the foundational step in the RLHF series.

2. **InstructionGPT:** Following the development of models using custom datasets, the next logical step might be the introduction of InstructionGPT, leveraging the insights gained from the initial custom dataset.

3. **SFT (Supervised Fine-tuning Trainer):** A tool that helps you easily adapt pre-trained models to specific tasks using labeled data, like teaching a language model to summarize news articles.

4. **PPO (Proximal Policy Optimization):** As reinforcement learning used to train in tasks where an agent interacts with an environment to learn optimal behaviors. PPO aims to improve stability and sample efficiency in comparison to other policy optimization methods by constraining the policy updates to prevent large policy changes during training.

4. **DPO (Direct Preference Optimization):** A stable and computationally lightweight algorithm for fine-tuning large-scale unsupervised language models, enabling precise control of their behavior by directly parameterizing the reward model and solving the reinforcement learning from human feedback problem with a simple classification loss, demonstrating effectiveness in aligning with human preferences across various tasks without the complexities of traditional RL methods.

5. **RRHF (Rank Responses to Align Language Models with Human Feedback without tears):** A simpler and more efficient alternative to traditional methods like Proximal Policy Optimization (PPO) for aligning large language models with human preferences, achieving comparable performance in alignment with PPO on the Helpful and Harmless dataset while requiring only 1 to 2 models and avoiding complex hyperparameter tuning.


This timeline highlights some key milestones, but it's crucial to remember that RLHF research is a rapidly evolving field with constant advancements. Many other relevant papers and projects exist, and the publication order can vary depending on specific subfields and applications.

For more specific information, I recommend searching for resources based on specific keywords or research areas within RLHF. Consider exploring databases like arXiv or Google Scholar with relevant queries like "reinforcement learning from human feedback," "human preferences in RL," or "policy learning with human guidance."

I hope this helps! Feel free to ask if you have any further questions about specific RLHF research or want to delve deeper into particular areas.


# Reference Resource & code
- https://vijayasriiyer.medium.com/rlhf-training-pipeline-for-llms-using-huggingface-821b76fc45c4