# Health Prediction System Using Facial Features

Ever since the outbreak of Coronavirus in late 2019, many things have
changed. COVID-19 is a disease that spreads very easily and quickly. In order to
control the spread of this disease, early diagnosis is immensely important to contain
the virus. Three of the most common early symptoms found in COVID-19 patients
are fever, sore throat and running nose. These symptoms have become the public’s
indicators in identifying anyone who is potentially infected by the disease. However,
it is tedious for one to detect if a group of people possess these symptoms on a case-
to-case basis. It is time-consuming and expensive for this task to be done manually
by humans.

Therefore, the idea of a health prediction system using facial features is
proposed to automate this process and increase its efficiency. The proposed health
prediction system takes facial images as input and predicts whether the person in the
image is healthy or ill with fever, sore throat or running nose. This project is
developed by using traditional machine learning approach. Four feature extraction
methods combined with four machine learning classifiers are experimented with and
evaluated to find the best model to be integrated into the health prediction system
user interface. The feature extraction methods used are Local Binary Pattern (LBP),
Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and
Gabor filter. The classifiers used are Support Vector Machine (SVM), Neural
Network (NN), k-Nearest Neighbours (KNN), and Random Forest (RF).

The best overall model chosen for the health prediction system user interface is the LBP+NN model, with the highest average testing accuracy of 76.84% in the second-level classification. It also performed considerably well in the first-level classification with lesser overfitting than the other models with similar performances, as it obtained 94.38% and 86.87% of average training and testing accuracies, respectively.
