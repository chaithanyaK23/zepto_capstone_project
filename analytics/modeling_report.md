# Q2 Part B: Modeling Report

## Class Balance and Split

|   survived |   count |   percent |
|-----------:|--------:|----------:|
|          0 |     549 |     61.75 |
|          1 |     340 |     38.25 |

A stratified train/test split is used before preprocessing so the survived/not-survived class ratio stays similar in both folds.

## Preprocessing

Numeric features are median-imputed and scaled with StandardScaler. Categorical features are most-frequent-imputed and one-hot encoded. These steps are inside scikit-learn Pipeline/ColumnTransformer objects, so they are fit only on training data and only transformed on test data.

## Classifier Comparison

| model               | confusion_matrix     |   accuracy |   precision |   recall |     f1 |    auc |
|:--------------------|:---------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression | [[98, 12], [21, 47]] |     0.8146 |      0.7966 |   0.6912 | 0.7402 | 0.861  |
| Decision Tree       | [[95, 15], [24, 44]] |     0.7809 |      0.7458 |   0.6471 | 0.6929 | 0.8573 |
| Random Forest       | [[94, 16], [21, 47]] |     0.7921 |      0.746  |   0.6912 | 0.7176 | 0.8373 |

## Imbalance Handling Comparison

| variant                      |   precision |   recall |     f1 |
|:-----------------------------|------------:|---------:|-------:|
| baseline_logistic_regression |      0.7966 |   0.6912 | 0.7402 |
| class_weight_balanced        |      0.7324 |   0.7647 | 0.7482 |
| smote_training_only          |      0.7463 |   0.7353 | 0.7407 |

The best imbalance strategy by F1 was class_weight_balanced with F1 0.7482. This comparison keeps SMOTE inside the training pipeline only, so synthetic examples are never created from the test fold.

## Random Forest GridSearchCV

- Best parameters: `{'classifier__max_depth': 5, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 200}`
- OOB score from best RandomForestClassifier(oob_score=True): 0.8158

## Regression Side Task

| model             |     MAE |    RMSE |     R2 |   Adjusted_R2 |
|:------------------|--------:|--------:|-------:|--------------:|
| Linear Regression | 21.0986 | 41.7021 | 0.3482 |        0.3091 |

The residual plot suggests heteroscedasticity because residual spread changes between lower and higher predicted fares.

## Final Model Comparison Table

Classification metrics and regression metrics are shown as separate metric groups because they are on different scales.

### Classification Metrics

| model               |   accuracy |   precision |   recall |     f1 |    auc | metric_group   |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|:---------------|
| Logistic Regression |     0.8146 |      0.7966 |   0.6912 | 0.7402 | 0.861  | classification |
| Decision Tree       |     0.7809 |      0.7458 |   0.6471 | 0.6929 | 0.8573 | classification |
| Random Forest       |     0.7921 |      0.746  |   0.6912 | 0.7176 | 0.8373 | classification |

### Regression Metrics

| model             |     MAE |    RMSE |     R2 |   Adjusted_R2 | metric_group   |
|:------------------|--------:|--------:|-------:|--------------:|:---------------|
| Linear Regression | 21.0986 | 41.7021 | 0.3482 |        0.3091 | regression     |

## Final Recommendation

I would deploy Logistic Regression because it produced the strongest F1 score (0.740) among the three baseline classifiers while also achieving accuracy 0.815, precision 0.797, recall 0.691, and AUC 0.861. F1 is a good primary metric here because the survived and not-survived classes are not perfectly balanced. The saved joblib artifact contains preprocessing plus the classifier, so it can predict from raw feature rows.

## Saved Pipeline Reload Check

- Saved artifact: `best_classifier_pipeline.joblib`
- Reloaded pipeline predictions on five raw rows: [0, 0, 0, 0, 0]
