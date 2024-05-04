from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def evaluate_trained_model(desired_labels, predicted_labels):
    confusion_matrix_res = confusion_matrix(desired_labels, predicted_labels)
    f1_score_res = f1_score(desired_labels, predicted_labels, average="micro")

    print("-------------------------- Evaluation of KNN performance --------------------------\n\n")
    print(f"Confusion Matrix : \n{confusion_matrix_res}")
    print(f"\n\n f1 Score for all classes: {f1_score_res}")
    print("\n\n-------------------------------------------------------------------------------\n\n")
