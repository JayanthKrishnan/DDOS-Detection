import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_model(df):
    """
    Function to process dataset, train an SVM model, and return accuracy score.
    """
    try:
        df.shape
        df.columns = df.columns.str.strip()
        df.Label.value_counts()
        df.isnull().sum().sum()

        #removing null values
        mask_netbios = (df['Label'] == 'NetBIOS') & (df.isnull().any(axis = 1))
        netbios_rows_with_missing = df[mask_netbios]
        df_cleaned = df.drop(netbios_rows_with_missing.index)
        df_cleaned.reset_index(drop=True, inplace=True)
        print('Shape of the dataset after removing rows with missing values that has label NetBIOS: ',df_cleaned.shape)
        df_cleaned.duplicated().sum()
        df_cleaned.Label.value_counts()

        #removing infinite values
        numeric_cols = df_cleaned.select_dtypes(include=[np.number])
        rows_with_inf = numeric_cols.apply(lambda x: np.isinf(x) | np.isneginf(x)).any(axis=1)
        num_rows_with_inf = rows_with_inf.sum()
        print("Number of rows with infinite value:",num_rows_with_inf)

        rows_with_inf = df_cleaned[rows_with_inf]
        df_cleaned = df_cleaned.drop(rows_with_inf.index)
        df_cleaned.reset_index(drop=True, inplace=True)
        print("Shape of dataset after removing rows with infinite values: ",df_cleaned.shape)
        df_cleaned.columns

        # checking if the columns "Fwd Header Length.1" and "Fwd Header Length" are same
        col1 = 'Fwd Header Length.1'
        col2 = 'Fwd Header Length'
        are_duplicates = (df_cleaned[col1] == df_cleaned[col2]).all()
        if are_duplicates:
            print(f"Columns {col1} and {col2} are identical.")
        else:
            print(f"Columns {col1} and {col2} are not identical.")
        df_cleaned = df_cleaned.drop(columns = ['Fwd Header Length.1'])
        df_cleaned.shape

        # bar plot
        label_counts = df_cleaned['Label'].value_counts()

        label_counts_df = label_counts.reset_index()
        label_counts_df.columns = ['Class Labels', 'Counts']

        plt.figure(figsize=(6, 4))
        sns.barplot(x='Class Labels', y='Counts', data=label_counts_df, palette='viridis', width=0.5)
        plt.xlabel('Class Labels')
        plt.ylabel('Counts')
        plt.title('Class Label Distribution')
        plt.show()
        df_cleaned.Label.value_counts()

        # Feature Selection /  drop columns that do not explain the target label
        columns_to_drop = ['Unnamed: 0', 'Flow ID', 'Destination IP', 'Source IP', 'Timestamp', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'SimillarHTTP']

        df_model = df_cleaned.drop(columns=columns_to_drop)
        df_model.shape

        # Outliers
        numeric_cols = df_model.select_dtypes(include='number').columns
        outliers_mask = pd.Series(False, index=df_model.index)

        for col in numeric_cols:
            Q1 = df_model[col].quantile(0.25)
            Q3 = df_model[col].quantile(0.75)
            IQR = Q3-Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask |= ((df_model[col] < lower_bound) | (df_model [col] > upper_bound)) & (df_model['Label'] == 'NetBIOS')
        plt.figure(figsize=(10,5))

        sns.boxplot(x=df_model['Label'], y=df_model['Flow Duration'], showfliers=False)
        sns.scatterplot(x=df_model['Label'][outliers_mask], y=df_model['Flow Duration'][outliers_mask], color='red', label='Outliers')

        plt.title('Box Plot of Flow Duraion with Outliers Highlighted')
        plt.xlabel('Label')
        plt.ylabel('Flow Duration')
        plt.legend()
        plt.show()
        df_model_clean = df_model[~outliers_mask]
        print("Original DataFrame shape:",df_model.shape)
        print("Clean DataFrame shape",df_model_clean.shape)
        df_model_clean.Label.value_counts()
        label_encoder = LabelEncoder()
        df_model_clean['Label'] = label_encoder.fit_transform(df_model_clean['Label'])
        x = df_model_clean.drop(columns=['Label'])
        y = df_model_clean['Label']

        rus = RandomUnderSampler(sampling_strategy=0.2)
        x, y = rus.fit_resample(x, y)
        print('After under sampling:')
        counts = pd.Series(y).value_counts()
        print("Counts of each class:")
        print(counts)
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        counts = pd.Series(Y_train).value_counts()
        print("Counts of each class:")
        print(counts)

        plt.figure(figsize=(6,4))
        sns.barplot(x=counts.index, y=counts.values, palette="viridis", width=0.5)
        plt.xlabel('Class Labels')
        plt.ylabel('Counts')
        plt.title('Value counts of Y_train')
        plt.show()
        smote = SMOTE(sampling_strategy=0.5)
        X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)
        print("After Under Sampling")
        counts = pd.Series(Y_train_balanced).value_counts()
        print("Counts of each class:")
        print(counts)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values, palette="viridis", width=0.5)
        plt.xlabel('Class Labels')
        plt.ylabel('Counts')
        plt.title('Value counts of Y_train')
        plt.show()

        scaler = StandardScaler()
        pca = PCA(n_components=0.95)
        pipeline = make_pipeline(scaler, pca)
        X_train_pca = pipeline.fit_transform(X_train_balanced)
        print(X_train_pca.shape, Y_train.shape)

        def evaluate_predictions(predictions, model):

            accuracy = accuracy_score(Y_test, predictions)

            weighted_precision = precision_score(Y_test, predictions, average='weighted', zero_division=0) 
            weighted_recall = recall_score(Y_test, predictions, average='weighted', zero_division=0) 
            f1 = f1_score(Y_test, predictions, average='weighted', zero_division=0)

            class_report = classification_report(Y_test, predictions, zero_division=0)

            # print(f'Evaluation Metrics for {model}:')
            # print(f"Accuracy: {accuracy * 100:.2f}%")/
            return accuracy
            # print(f"Weighted Precision: {weighted_precision*100:.2f}%") 
            # print(f"Weighted Recall: {weighted_recall*100:.2f}%")
            # print(f"F1 Score: {f1*100:.2f}%")
            # print("\nClassification Report:\n")
            # print(class_report)


        svm = SVC(kernel='linear', C=0.1, random_state=42)
        # Fit the SVM model on the training data 
        svm.fit(X_train_pca, Y_train_balanced)

        #Predicting the test set results

        y_pred_svm = svm.predict(pipeline.transform(X_test))
        conf_matrix = confusion_matrix(Y_test, y_pred_svm)
        print("Confusion Matrix:\n", conf_matrix)

        #Evaluating the model
        # print(y_pred_svm)
        ac = evaluate_predictions (y_pred_svm, model = 'SVM')
        return round(ac*100,2)
    
    except Exception as e:
        return str(e)

# Example usage (for testing purpose)
if __name__ == "__main__":
    df = pd.read_csv("NetBIOS-testing.csv")  # Replace with actual dataset
    accuracy = evaluate_model(df)
    print(f"Accuracy: {accuracy * 100:.2f}%")
