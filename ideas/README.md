

# Ideas


## Prompt Engineering idea

- Inspired by [this video here](https://youtu.be/bZQun8Y4L2A?si=H67ir3WCFdczydK4&t=1492), maybe we can generate text before sending prompt to the "evaluator". In this way, we can have like steps like this

    - step 1: generate a prompt that sounds something like this:
    ```
    You are a university professor, renowned as an expert in your field. Your teaching style is known for providing comprehensive and lengthy explanations, ensuring that your students grasp the depths of the topic at hand.
    
    Context: A student of yours approaches you after class, presenting a query that's been puzzling her. She's received varied opinions from her peers and is seeking clarity.
    
    Student's Question: 
    
    Q
    
    She's been given the list of potential answers could be:
    Option 1: option_1
    Option 2: option_2
    Option 3: option_3
    Option 4: option_4
    Option 5: option_5

    Your student might be right, but she might also be wrong.
    Kindly delve into the problem, elucidating the validity of her answer. Moreover, critically examine each option, helping her discern whether the answer that she chose is correct or wrong. Provide a step-by-step analysis in this way:

    Review of option 1: (write here your review of option 1)
    Review of option 2: (write here your review of option 2)
    Review of option 3: (write here your review of option 3)
    Review of option 4: (write here your review of option 4)
    Review of option 5: (write here your review of option 5)


    ```
    - step 2: send this prompt to the llama/deberta and train a usual classifier. Then it makes sense for this classifier to give probability 0 for an answer, if it sees a better answer in the prompt options.
