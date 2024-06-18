For decision_trees_medicine.py:

Using decision tree algorithms, I predict which medication patients respond to based on their characteristics (multiclass classifier).

The table drug200.csv contains data about a set of patients, together with the 5 medications they reacted to: Drug A, Drug B, Drug C, Drug x and Drug y.

With decision trees, I build a model to find out which drug might be appropriate for a future patient. 
I also assess the model accuracy and visualize the decision tree with graphviz.

# Notice: You might need to install the pydotplus and graphviz libraries if you have not installed these before:
conda install -c conda-forge pydotplus -y
conda install -c conda-forge python-graphviz -y



For decision_trees_music.py:

Using decision tree algorithms, I predict which music genre individuals like based on age and gender.

The table music.csv contains the gender and age of several individuals, which is used to guess which music genre they prefer.

With decision trees, I will train the model on a training dataset and I will then predict which type of music new people like, using a test dataset. 
I assess the model accuracy and save the predictive model using joblib. I then visualize the decision tree with graphviz.
