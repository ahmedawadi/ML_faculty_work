from preparation import prepare_data as pd
from models import train_model as tm
from models import evaluate_model as em 

#prepration of the data for the training phase
features_to_train,features_to_test, labels_to_train, labels_to_test = pd.prepare_data_for_training()

'''start the first training with the knn model'''
knn = tm.train_with_knn(features=features_to_train, labels=labels_to_train, neighbors=5)

#checking the confusion matrix and f_score of our mode
predicted_labels = knn.predict(features_to_test)
em.evaluate_trained_model(labels_to_test, predicted_labels, algorithm="KNN")

'''
start the second training with the MLPClassifier model 
use relu function as an activation function 
hidden layers parameters are choosed by default 
'''
mlp_classifer = tm.train_with_MLPClassifier(features=features_to_train, labels=labels_to_train, hidden_layers=(50, 30, 10), activation_function='relu')

#evaluation of the MLP classifier model
predicted_labels = mlp_classifer.predict(features_to_test)
em.evaluate_trained_model(labels_to_test, predicted_labels, algorithm="MLP Classifier")

'''
Start the thrid training with SVM model 
best params will choosed using the grid search
'''
#choosing best params for our svm model
best_svm_params = tm.get_svc_best_params(features_to_train, labels_to_train)

#training SVM model with different kernels to see the difference
kernels = ["sigmoid", "poly", "rbf"]

for kernel in kernels:
    svm_model = tm.train_data_with_SVM_model_using_poly_kernel(features_to_train, labels_to_train, C=best_svm_params.get("C"), gamma=best_svm_params.get("gamma"), kernel_=kernel)
    
    #evaluation of the SVM model with a specific kernel
    predicted_labels = svm_model.predict(features_to_test)
    em.evaluate_trained_model(labels_to_test, predicted_labels, algorithm=f"SVM with {kernel} kernel")