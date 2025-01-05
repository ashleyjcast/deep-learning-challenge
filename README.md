# deep-learning-challenge
Module 21 Challenge 
Alphabet Soup Charity Analysis


Overview:
The purpose of this analysis was to develop a binary classification model for Alphabet Soup, a nonprofit organization. The model was designed to predict the probability of success for funding applicants based on various components of the dataset. This analysis helps Alphabet Soupmake data-driven decisions when allocating funds to applicants with the best potential for success.


Project Objectives:

- Preprocess the dataset to prepare it for machine learning modeling.

- Build a neural network using TensorFlow and Keras to classify applicants as successful or unsuccessful.

- Optimize the neural network model by adjusting hyperparameters and architecture to improve performance.

- Evaluate the model's performance and provide recommendations for future improvements.


Technologies Used:

- Python 
- TensorFlow
- Keras
- Pandas
- scikit-learn
- Google Colab


Dataset Description:

The dataset contains information about organizations that have applied for funding. Key columns include:

- EIN: Employer Identification Number (dropped, irrelevant)

- NAME: Organization name (dropped, irrelevant)

- APPLICATION_TYPE: Type of application submitted

- AFFILIATION: Organization’s affiliation type

- CLASSIFICATION: Government classification of the organization

- USE_CASE: Funding use case 

- ORGANIZATION: Organization type 

- STATUS: Organization's operational status (dropped, low variance)

- INCOME_AMT: Income classification amount

- ASK_AMT: Funding amount requested

- SPECIAL_CONSIDERATIONS: Whether special considerations are flagged (dropped, low variance)

- IS_SUCCESSFUL: Binary variable indicating funding success (1 = successful, 0 = unsuccessful) (target variable)
  


Steps in Analysis

1. Data Preprocessing:

  - Removed Columns: EIN, NAME, STATUS, SPECIAL_CONSIDERATIONS (irrelevant or low variance).

  - Encoded Categorical Variables: Applied one-hot encoding to columns APPLICATION_TYPE and CLASSIFICATION
  - Scaled Numerical Data: Applied StandardScaler to scale all feature data, ensuring consistent feature contribution by normalizing X_train and X_test.


2. Building the Neural Network:

Base Model:

- Two hidden layers:

1st Layer: 80 neurons, ReLU activation.

2nd Layer: 30 neurons, ReLU activation.

- Output Layer:

1 neuron, Sigmoid activation (binary classification).

- Accuracy: 72.83% | Loss: 0.5660
  

Optimization Attempts:

- Attempt 1:

Added a 3rd hidden layer with 25 neurons.

Accuracy: 72.90% | Loss: 0.5716


- Attempt 2:

Increased neurons in all layers:

1st Layer: 200 neurons.

2nd Layer: 150 neurons.

3rd Layer: 100 neurons.

Accuracy: 73.01% | Loss: 0.6055


- Attempt 3:

Added a 4th hidden layer:

1st Layer: 200 neurons.

2nd Layer: 150 neurons.

3rd Layer: 100 neurons.

4th Layer: 50 neurons.

Accuracy: 73.07% | Loss: 0.6191


Results and Recommendations

- Key Findings:
The best accuracy achieved was 73.07% with the third optimization attempt. In this attempt two additional hidden layers were added, and neurons were progressively increased throughout earlier attempts to reach this outcome.

Loss metrics suggest limitations in predictability, leading to further complexity.

Target model performance of 75% was not achieved. 

  
- Recommendations: 
  Conduct feature engineering to extract additional meaningful features or explore external datasets to enhance predictive power

  Explore additional data sources or clean noisy data to reduce loss.


Conclusion:
The neural network achieved a maximum accuracy of 73.07%. While the performance improvements were marginal across attempts, this model provides a solid foundation for predicting funding success. Future efforts should focus on developing feature and testing alternative models to further enhance accuracy.
