import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_absolute_percentage_error
from catboost import CatBoostRegressor
import shap
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.regressor import ResidualsPlot
import matplotlib.pyplot as plt
import yaml
import pickle


def train_model(config_path="./config/config.yml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Getting data")
    df = pd.read_csv("./data/model_input.csv")

    numerical_features = config["model_input"]["numerical_features"]
    categorical_features = config["model_input"]["categorical_features"]
    target = config["model_input"]["target"]

    scaler = MinMaxScaler()
    data_pipeline = ColumnTransformer(
        [("numerical", scaler, numerical_features)], remainder="passthrough"
    )

    print("Train test split")
    X = df[numerical_features + categorical_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12345
    )

    print("Scaling data")
    data_pipeline.fit(X_train)

    with open("./model/data_pipeline.pkl", "wb") as f:
        pickle.dump(data_pipeline, f)

    X_train_transformed = data_pipeline.transform(X_train)
    X_test_transformed = data_pipeline.transform(X_test)
    X_train_transformed = pd.DataFrame(
        X_train_transformed, columns=numerical_features + categorical_features
    )
    X_test_transformed = pd.DataFrame(
        X_test_transformed, columns=numerical_features + categorical_features
    )

    print("Training model")
    if config["model_training"]["use_grid_search"]:
        print("Starting grid search")
        param_grid = config["model_training"]["grid_search"]["param_grid"]
        mape_scorer = make_scorer(
            mean_absolute_percentage_error, greater_is_better=False
        )

        grid_search = GridSearchCV(
            CatBoostRegressor(
                cat_features=categorical_features, eval_metric="MAPE", verbose=False
            ),
            param_grid=param_grid,
            cv=3,
            return_train_score=True,
            n_jobs=-1,
            scoring=mape_scorer,
            verbose=True,
        )
        grid_search.fit(X_train_transformed, y_train)
        pd.DataFrame(grid_search.cv_results_).sort_values("rank_test_score").to_csv(
            "./report/cv_results.csv", index=False
        )
        best_params = grid_search.best_params_
    else:
        print("Skipping grid search")
        best_params = config["model_training"]["without_grid_search"]["best_params"]

    print("Using params:")
    [print(f"{w}: {v}") for w, v in best_params.items()]
    model = CatBoostRegressor(
        cat_features=categorical_features,
        eval_metric="MAPE",
        use_best_model=True,
        **best_params,
    )
    model.fit(
        X_train_transformed,
        y_train,
        eval_set=(X_test_transformed, y_test),
        verbose=False,
    )

    print("Testing model and saving reports")
    y_pred = model.predict(X_test_transformed)
    y_pred_train = model.predict(X_train_transformed)

    mape_catboost = mean_absolute_percentage_error(y_test, y_pred)
    r2_catboost = r2_score(y_test, y_pred)
    mape_catboost_train = mean_absolute_percentage_error(y_train, y_pred_train)
    r2_catboost_train = r2_score(y_train, y_pred_train)

    print(f"Training mape: {mape_catboost_train}, test mape: {mape_catboost}")
    print(f"Training r2: {r2_catboost_train}, test r2: {r2_catboost}")
    wrapped_model = wrap(model)
    visualizer = ResidualsPlot(wrapped_model)

    visualizer.fit(
        X_train_transformed, y_train
    )  # Fit the training data to the visualizer
    visualizer.score(X_test_transformed, y_test)  # Evaluate the model on the test data
    visualizer.show(outpath="./report/residuals.png")
    plt.clf()

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train_transformed)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("./report/shap.png")

    model.get_feature_importance(prettified=True).to_csv(
        "./report/feature_importance.csv", index=False
    )

    print("Saving model")
    model.save_model("./model/model.cbm")
    print("Done")
    return {
        "mape_test": mape_catboost,
        "mape_train": mape_catboost_train,
        "r2_test": r2_catboost,
        "r2_train": r2_catboost_train,
    }
