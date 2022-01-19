import wandb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

WANDB_PROJECT_NAME = "hyperparameter-optimization"

with wandb.init(project=WANDB_PROJECT_NAME):
    df = pd.read_csv('data\heart.csv')
    X = df.drop(['target'], axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    config = wandb.config
    rfc = RandomForestClassifier(
        bootstrap=config.bootstrap,
        max_depth = config.max_depth,
        max_features = config.max_features,
        min_samples_leaf = config.min_samples_leaf,
        min_samples_split = config.min_samples_split,
        n_estimators = config.n_estimators,
    )

    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    wandb.log({'accuracy': accuracy_score(y_test, y_pred)})
