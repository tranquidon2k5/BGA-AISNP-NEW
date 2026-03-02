# src/models.py
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def make_xgb_multiclass(num_classes: int, **kwargs):
    """
    Tạo XGBClassifier cho multi-class ancestry.
    kwargs dùng để override hyperparams nếu cần (tuning).
    """
    default_params = dict(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,  # để reproducible
    )
    default_params.update(kwargs)
    model = xgb.XGBClassifier(**default_params)
    return model


def train_and_eval(model, X_train, y_train, X_test, y_test, label_encoder=None, title=""):
    """
    Train model và in ra accuracy + classification_report + confusion_matrix.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"=== {title} ===")
    print(f"Accuracy: {acc:.4f}")

    if label_encoder is not None:
        target_names = label_encoder.classes_
    else:
        target_names = None

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            digits=4,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return acc, y_pred
