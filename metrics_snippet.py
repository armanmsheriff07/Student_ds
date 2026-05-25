import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


def run_metrics():
    data = pd.read_csv("student_data.csv")
    X = data[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Score']]
    y = data['Pass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    if probs is not None:
        try:
            print("ROC AUC:", roc_auc_score(y_test, probs))
        except Exception as e:
            print("ROC AUC could not be computed:", e)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("5-fold CV accuracy: mean=%.4f std=%.4f" % (cv_scores.mean(), cv_scores.std()))


if __name__ == '__main__':
    run_metrics()
