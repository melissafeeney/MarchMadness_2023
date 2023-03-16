# MarchMadness_2023
# Designing a bracket for the 2023 Men's NCAA March Madness Tournament using deep learning

**It’s March again! Can a deep learning algorithm design a successful 2023 March Madness tournament bracket?**

With the volume of college basketball data available, it can be overwhelming to determine what is most important, and recognize patterns that are indicative of teams’ predicted performances. Enter machine learning, specifically deep learning in this case, which can be trained for this task.

Last year was my first year using an algorithm to build a bracket, and there were things that I wanted to do differently this year. One was creating variables to represent the relative strength of each team and their schedule- something to differentiate teams winning lots of games in less competitive conferences, versus teams winning fewer numbers of games in very competitive conferences. A model trained without this would likely not be able to make this distinction, because to an algorithm, a win is a win. As a proxy, I integrated Ken Pomeroy rankings into my model, considering the average, best, and worst ranking for each team, each season. This established a confidence band with upper, average, and lower bounds in the relative ranking of each team each season. I combined these ranking variables with the number of wins and losses, along with ratios representing % of field goals made, % of 3-pointers made, and assists to turnovers. I trained my model on every game from 2003 through 2023 (pre-tournament), and my intention was to encourage the model to identify recurrent patterns in teams’ seasonal statistics that made them more or less likely to win a game, that may not be obvious to humans. 

I developed a neural network and used KerasTuner with Bayesian optimization to determine optimal hyperparameters- the number of dense layers, number of hidden units, and use of batch normalization for regularization. The architecture of the optimal model included 11 dense layers, each with a leaky ReLU activation function and alpha of 0.01, and containing hidden unit counts of 30 to 500. 

## Network Architecture:

![Screen Shot 2023-03-16 at 12 21 23 PM](https://user-images.githubusercontent.com/31778500/225685709-3071fd73-4c6c-4579-9d30-300940a86aee.png)

## Training Results:

I employed a 70/30 training/test split on games from the 2003 season through the 2023 regular season, and yielded an accuracy of 76%, a ROC AUC of 0.84, an F1 score of 0.76, a precision score of 0.74, and a recall score of 0.78 on the test set data. 

![Screen Shot 2023-03-16 at 12 24 39 PM](https://user-images.githubusercontent.com/31778500/225686583-df0415ca-72f9-42a0-ae55-71afffc667c7.png)

Considering the prediction probabilities of each game, at the start of the tournament, the model is more certain, with many probabilities above 90%. But as the tournament progresses, the probabilities move closer to 50%- implying that the model recognizes that teams should be more evenly matched in later games.

## Model-produced Bracket with game-by-game prediction probabilities:

![Model_Output_Bracket](https://user-images.githubusercontent.com/31778500/225686922-7735412c-a8df-4d03-a840-c295393c0fce.png)

