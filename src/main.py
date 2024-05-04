from preparation import prepare_data as pd
from models import train_model as tm
from models import evaluate_model as em 

#prepration of the data for the training phase
features_to_train,features_to_test, labels_to_train, labels_to_test = pd.prepare_data_for_training()

#start the first training with the knn model
knn = tm.train_with_knn(features=features_to_train, labels=labels_to_train, neighbors=5)

#checking the confusion matrix and f_score of our mode
predicted_labels = knn.predict(features_to_test)
em.evaluate_trained_model(labels_to_test, predicted_labels)