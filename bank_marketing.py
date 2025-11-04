# ---------------------------------------------
# Bank Marketing Dataset - Decision Tree Classifier
# ---------------------------------------------

import zipfile
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------
# 1️⃣ Extract the Main ZIP File
# ---------------------------------------------
main_zip = "bank_marketing.zip"
extract_dir = "bank_data"

with zipfile.ZipFile(main_zip, "r") as main_ref:
    main_ref.extractall(extract_dir)

# Now extract the nested zip (bank.zip)
nested_zip_path = os.path.join(extract_dir, "bank.zip")

if not os.path.exists(nested_zip_path):
    raise FileNotFoundError("❌ 'bank.zip' not found inside the main zip!")

nested_extract_dir = os.path.join(extract_dir, "bank")
with zipfile.ZipFile(nested_zip_path, "r") as nested_ref:
    nested_ref.extractall(nested_extract_dir)

# ---------------------------------------------
# 2️⃣ Locate CSV File and Load Dataset
# ---------------------------------------------
csv_file = None
for root, dirs, files in os.walk(nested_extract_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_file = os.path.join(root, file)
            break

if not csv_file:
    raise FileNotFoundError("❌ No CSV file found even after extracting nested zips.")

print(f"✅ Found CSV: {csv_file}")

# The bank dataset uses semicolon (;) as separator
df = pd.read_csv(csv_file, sep=';')
print("\n✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print("\nPreview:")
print(df.head())

# ---------------------------------------------
# 3️⃣ Data Preprocessing
# ---------------------------------------------
print("\n--- Checking Missing Values ---")
print(df.isnull().sum())

# Convert categorical columns to numerical
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop('y_yes', axis=1)
y = df['y_yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# 4️⃣ Decision Tree Model
# ---------------------------------------------
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# ---------------------------------------------
# 5️⃣ Evaluation
# ---------------------------------------------
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------
# 6️⃣ Visualization
# ---------------------------------------------


clf_simple = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=12, random_state=42)
clf_simple.fit(X_train, y_train)

plt.figure(figsize=(26, 15))  # smaller figure
plot_tree(
    clf_simple,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10,     # smaller font
    proportion=False # avoids stretching horizontally
)

plt.title("Simplified Decision Tree  (Depth = 4)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()


# Feature importance plot
plt.figure("Feature Importance", figsize=(10, 6))
feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
sns.barplot(x=feat_imp, y=feat_imp.index, palette='viridis')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
