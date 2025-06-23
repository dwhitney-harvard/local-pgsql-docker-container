import pandas as pd
import xgboost as xgb
import joblib
import time
from extract_features import extract_features

# Load training pairs
start_time = time.time()
df = pd.read_csv("training_pairs_expanded.csv")
print("✅ Finished reading training_pairs_expanded.csv into DataFrame in %s seconds" % (time.time() - start_time))

# Extract features and labels
t1 = time.time()
X, y = extract_features(df)
print("✅ Finished feature extraction in %s seconds" % (time.time() - t1))

# Train model
t2 = time.time()
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    # use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X, y)
print("✅ Finished training model in %s seconds" % (time.time() - t2))

# Save model
t3 = time.time()
joblib.dump(model, "dedup_model.pkl")
print("✅ Finished saving dedup_model.pkl in %s seconds" % (time.time() - t3))

print("✅ Finished training and saving model in %s seconds" % (time.time() - start_time))
