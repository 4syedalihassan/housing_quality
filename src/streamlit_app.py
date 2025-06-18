# 333app.py
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandera as pa
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import warnings

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(page_title="Housing Quality Prediction Pipeline", layout="wide")
st.title("🏠 Advanced Housing Quality Assessment & Prediction Pipeline")

# Enhanced Schema
schema = pa.DataFrameSchema(
    columns={
        "YearBuilt": pa.Column(int, pa.Check.ge(1800)),
        "YearRemodAdd": pa.Column(int),
        "OverallQual": pa.Column(int, pa.Check.isin(range(1, 11))),
        "OverallCond": pa.Column(int, pa.Check.isin(range(1, 11))),
        "Garage": pa.Column(str, nullable=False),
        "Area": pa.Column(float, pa.Check.gt(0)),
        "HouseStyle": pa.Column(str),
        "BldgType": pa.Column(str),
        "BedroomAbvGr": pa.Column(int, pa.Check.ge(0)),
        "KitchenAbvGr": pa.Column(int, pa.Check.ge(0))
    }
)


def load_data(uploaded_file):
    """Load and combine all Excel sheets with duplicate handling"""
    with st.spinner("Loading data..."):
        xls = pd.ExcelFile(uploaded_file)
        sheets = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            sheets.append(df)
        combined_df = pd.concat(sheets, ignore_index=True)
        combined_df = combined_df.dropna(how='all')
        return combined_df


def parse_garage_features(df):
    """Enhanced garage feature extraction"""
    garage_features = pd.DataFrame()

    # Basic garage type encoding
    garage_mapping = {
        "Attchd": 3, "Detchd": 2, "BuiltIn": 4,
        "Basement": 1, "2Types": 5, "Unknown": 0
    }
    garage_features['garage_type_encoded'] = df['Garage'].map(garage_mapping)

    # Garage quality score based on type
    garage_features['garage_quality_score'] = np.where(
        df['Garage'].isin(['BuiltIn', 'Attchd']), 3,
        np.where(df['Garage'].isin(['Detchd', '2Types']), 2,
                 np.where(df['Garage'] == 'Basement', 1, 0))
    )

    # Create garage age feature
    garage_features['garage_age'] = 2025 - df['YearBuilt']

    return garage_features


def encode_categorical_features(df):
    """Comprehensive categorical feature encoding"""
    encoded_df = df.copy()
    categorical_features = {}

    # Ensure no NaN values in categorical columns before encoding
    categorical_columns = ['HouseStyle', 'LandContour', 'BldgType', 'Garage', 'Location']

    for col in categorical_columns:
        if col in encoded_df.columns:
            # Convert to string and handle NaN/None values
            encoded_df[col] = encoded_df[col].astype(str).replace(['nan', 'None', 'NaN'], 'Unknown')
            encoded_df[col] = encoded_df[col].fillna('Unknown')

    # One-hot encoding for nominal categories
    nominal_cols = ['HouseStyle', 'LandContour', 'BldgType', 'Location']
    for col in nominal_cols:
        if col in encoded_df.columns:
            try:
                dummies = pd.get_dummies(encoded_df[col], prefix=col, drop_first=True)
                encoded_df = pd.concat([encoded_df, dummies], axis=1)
                categorical_features[col] = list(dummies.columns)
            except Exception as e:
                st.warning(f"Could not encode {col}: {e}")

    # Label encoding for ordinal categories (Garage)
    if 'Garage' in encoded_df.columns:
        try:
            garage_mapping = {
                "Attchd": 3, "Detchd": 2, "BuiltIn": 4,
                "Basement": 1, "2Types": 5, "Unknown": 0
            }
            encoded_df['Garage_encoded'] = encoded_df['Garage'].map(garage_mapping).fillna(0)
            categorical_features['Garage'] = 'label_encoded'
        except Exception as e:
            st.warning(f"Could not encode Garage: {e}")

    # Add garage features
    try:
        garage_features = parse_garage_features(df)
        encoded_df = pd.concat([encoded_df, garage_features], axis=1)
    except Exception as e:
        st.warning(f"Could not create garage features: {e}")

    return encoded_df, categorical_features


def explore_dataset(df):
    """Comprehensive dataset exploration focusing on quality relationships"""
    st.subheader("📊 Dataset Exploration & Quality Analysis")

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Avg. House Age", f"{(2025 - df['YearBuilt'].mean()):.0f} years")
    with col4:
        st.metric("Avg. Quality Rating", f"{df['OverallQual'].mean():.1f}/10")

    # Quality Distribution Analysis
    st.subheader("🔍 Key Finding #1: Quality vs Condition Relationship")

    col1, col2 = st.columns(2)

    with col1:
        # Quality distribution
        quality_counts = df['OverallQual'].value_counts().sort_index()
        fig1 = px.bar(x=quality_counts.index, y=quality_counts.values,
                      title='Distribution of Overall Quality Ratings',
                      labels={'x': 'Quality Rating', 'y': 'Number of Properties'})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Quality vs Condition scatter
        fig2 = px.scatter(df, x='OverallQual', y='OverallCond',
                          title='Quality vs Condition Relationship',
                          labels={'OverallQual': 'Overall Quality', 'OverallCond': 'Overall Condition'})
        st.plotly_chart(fig2, use_container_width=True)

    qual_cond_corr = df['OverallQual'].corr(df['OverallCond'])
    st.write(f"**Insight**: Correlation between quality and condition: {qual_cond_corr:.3f}")
    st.write(
        "**Interpretation**: Quality and condition ratings show moderate correlation, indicating they capture different aspects of property assessment.")

    # Finding 2: Age Impact on Quality
    st.subheader("🔍 Key Finding #2: House Age Impact on Quality Assessment")

    df['HouseAge'] = 2025 - df['YearBuilt']
    df['AgeGroup'] = pd.cut(df['HouseAge'],
                            bins=[0, 10, 25, 50, 100],
                            labels=['New (0-10)', 'Modern (11-25)', 'Mature (26-50)', 'Old (50+)'])

    age_quality = df.groupby('AgeGroup')['OverallQual'].agg(['mean', 'std']).reset_index()

    fig3 = px.box(df, x='AgeGroup', y='OverallQual',
                  title='Quality Ratings by House Age Group')
    st.plotly_chart(fig3, use_container_width=True)

    age_quality_corr = df['HouseAge'].corr(df['OverallQual'])
    st.write(f"**Insight**: Correlation between house age and quality: {age_quality_corr:.3f}")
    st.write(
        "**Interpretation**: Newer properties tend to have higher quality ratings, but well-maintained older properties can also achieve high ratings.")

    # Finding 3: Property Features Impact on Quality
    st.subheader("🔍 Key Finding #3: Property Features vs Quality")

    col1, col2 = st.columns(2)

    with col1:
        # Area vs Quality
        fig4 = px.scatter(df, x='Area', y='OverallQual', color='BldgType',
                          title='Living Area vs Quality by Building Type',
                          hover_data=['YearBuilt', 'HouseStyle'])
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        # Garage type vs Quality
        garage_quality = df.groupby('Garage')['OverallQual'].mean().sort_values(ascending=False)
        fig5 = px.bar(x=garage_quality.index, y=garage_quality.values,
                      title='Average Quality by Garage Type')
        fig5.update_layout(xaxis_title='Garage Type', yaxis_title='Average Quality Rating')
        st.plotly_chart(fig5, use_container_width=True)

    area_quality_corr = df['Area'].corr(df['OverallQual'])
    st.write(
        f"**Insight**: Area-Quality correlation: {area_quality_corr:.3f}, Garage type significantly impacts quality perception.")

    # Bedroom/Kitchen Analysis
    st.subheader("🔍 Key Finding #4: Room Configuration Impact")

    col1, col2 = st.columns(2)

    with col1:
        bedroom_quality = df.groupby('BedroomAbvGr')['OverallQual'].mean()
        fig6 = px.line(x=bedroom_quality.index, y=bedroom_quality.values,
                       title='Average Quality by Number of Bedrooms',
                       markers=True)
        fig6.update_layout(xaxis_title='Bedrooms Above Ground', yaxis_title='Average Quality')
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        kitchen_quality = df.groupby('KitchenAbvGr')['OverallQual'].mean()
        fig7 = px.bar(x=kitchen_quality.index, y=kitchen_quality.values,
                      title='Average Quality by Number of Kitchens')
        fig7.update_layout(xaxis_title='Kitchens Above Ground', yaxis_title='Average Quality')
        st.plotly_chart(fig7, use_container_width=True)

    # Feature Correlation Heatmap
    st.subheader("🌡️ Feature Correlation Analysis")
    numeric_cols = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'Area', 'BedroomAbvGr', 'KitchenAbvGr']
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_numeric_cols) > 3:
        corr_matrix = df[available_numeric_cols].corr()
        fig8 = px.imshow(corr_matrix,
                         title='Feature Correlation Heatmap',
                         color_continuous_scale='RdBu_r',
                         aspect='auto')
        fig8.update_layout(height=500)
        st.plotly_chart(fig8, use_container_width=True)

    return {
        'quality_condition_correlation': qual_cond_corr,
        'age_quality_correlation': age_quality_corr,
        'area_quality_correlation': area_quality_corr,
        'garage_quality_impact': garage_quality.to_dict(),
        'key_insights': [
            f"Quality and condition ratings have {qual_cond_corr:.2f} correlation",
            f"House age shows {age_quality_corr:.2f} correlation with quality",
            f"Living area correlates {area_quality_corr:.2f} with quality ratings",
            "Garage type significantly impacts perceived quality",
            "Room configuration affects quality assessment patterns"
        ]
    }


def build_quality_prediction_model(df):
    """Build and evaluate quality prediction models with robust error handling"""
    st.subheader("🤖 Quality Prediction Modeling")

    # Initial data validation
    st.write(f"**Initial Dataset**: {len(df)} rows, {len(df.columns)} columns")

    if len(df) == 0:
        st.error("❌ No data available for modeling. Please check your data upload.")
        return None, None, None, None

    # Choose prediction target
    available_targets = []
    if 'OverallQual' in df.columns:
        available_targets.append('OverallQual')
    if 'OverallCond' in df.columns:
        available_targets.append('OverallCond')

    if not available_targets:
        st.error("❌ Neither 'OverallQual' nor 'OverallCond' columns found in the dataset.")
        return None, None, None, None

    target_choice = st.selectbox(
        "Select Prediction Target:",
        available_targets,
        help="Choose whether to predict Overall Quality or Overall Condition"
    )

    # Check target variable
    target_series = df[target_choice].dropna()
    if len(target_series) == 0:
        st.error(f"❌ No valid values found in {target_choice} column.")
        return None, None, None, None

    st.write(f"**Target Variable Info**: {target_choice} has {len(target_series)} valid values")
    st.write(f"**Target Range**: {target_series.min()} to {target_series.max()}")

    # Feature engineering with better error handling
    try:
        encoded_df, cat_features = encode_categorical_features(df)
        st.write(f"**After Encoding**: {len(encoded_df)} rows, {len(encoded_df.columns)} columns")
    except Exception as e:
        st.warning(f"⚠️ Error in categorical encoding: {e}")
        encoded_df = df.copy()
        cat_features = {}

    # Define base feature columns that are most likely to exist
    base_features = ['Area', 'YearBuilt', 'YearRemodAdd', 'BedroomAbvGr', 'KitchenAbvGr']
    feature_cols = []

    # Add base features if they exist
    for col in base_features:
        if col in encoded_df.columns:
            feature_cols.append(col)

    # Add garage features if available
    garage_features = ['garage_type_encoded', 'garage_quality_score', 'garage_age']
    for col in garage_features:
        if col in encoded_df.columns:
            feature_cols.append(col)

    # Add the other quality measure as a feature if available
    if target_choice == 'OverallQual' and 'OverallCond' in encoded_df.columns:
        feature_cols.append('OverallCond')
    elif target_choice == 'OverallCond' and 'OverallQual' in encoded_df.columns:
        feature_cols.append('OverallQual')

    # Add encoded categorical features
    for col in encoded_df.columns:
        if any(cat in col for cat in ['HouseStyle_', 'LandContour_', 'BldgType_', 'Location_']):
            feature_cols.append(col)

    # Remove duplicates and ensure features exist
    available_features = list(set([col for col in feature_cols if col in encoded_df.columns]))

    if not available_features:
        st.error("❌ No suitable features found for modeling.")
        st.write("**Available columns:**", list(encoded_df.columns))
        return None, None, None, None

    st.write(
        f"**Selected Features ({len(available_features)})**: {', '.join(available_features[:10])}{'...' if len(available_features) > 10 else ''}")

    # Prepare data with robust preprocessing
    try:
        X = encoded_df[available_features].copy()
        y = encoded_df[target_choice].copy()

        st.write(f"**Before Preprocessing**: X shape {X.shape}, y shape {y.shape}")

        # Remove rows where target is missing
        valid_target_mask = ~y.isna()
        X = X[valid_target_mask]
        y = y[valid_target_mask]

        st.write(f"**After Target Filtering**: X shape {X.shape}, y shape {y.shape}")

        if len(X) == 0:
            st.error("❌ No valid samples after removing missing target values.")
            return None, None, None, None

        # Handle missing values in features
        st.write(f"**Missing Values Check**: {X.isna().sum().sum()} missing values found")

        # Fill missing values more robustly
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                # Use median for numeric columns
                median_val = X[col].median()
                if pd.isna(median_val):  # If all values are NaN
                    median_val = 0
                X[col] = X[col].fillna(median_val)
            else:
                # Use mode for categorical columns
                mode_val = X[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                X[col] = X[col].fillna(fill_val)

        # Convert all columns to numeric where possible
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(0)
                except:
                    # If conversion fails, use label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

        # Final check for any remaining NaN values
        X = X.fillna(0)
        y = y.fillna(y.median())

        st.write(f"**After Preprocessing**: X shape {X.shape}, y shape {y.shape}")
        st.write(f"**Remaining Missing Values**: X: {X.isna().sum().sum()}, y: {y.isna().sum()}")

    except Exception as e:
        st.error(f"❌ Error in data preprocessing: {e}")
        return None, None, None, None

    # Create classification problem with better binning
    try:
        # Check the actual range of target values
        y_min, y_max = y.min(), y.max()
        st.write(f"**Target Value Range**: {y_min} to {y_max}")

        # Create bins based on actual data distribution
        if y_max <= 10:  # Assuming quality scale 1-10
            # Use quantile-based binning for better distribution
            bins = [0, y.quantile(0.25), y.quantile(0.5), y.quantile(0.75), y_max + 0.1]
            labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
        else:
            # Use equal-width binning
            bin_width = (y_max - y_min) / 4
            bins = [y_min - 0.1, y_min + bin_width, y_min + 2 * bin_width, y_min + 3 * bin_width, y_max + 0.1]
            labels = ['Low', 'Medium-Low', 'Medium-High', 'High']

        y_class = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

        # Remove any NaN values created by binning
        valid_class_mask = ~y_class.isna()
        X = X[valid_class_mask]
        y_class = y_class[valid_class_mask]

        st.write(f"**After Classification Binning**: {len(X)} samples")
        st.write(f"**Class Distribution**:")
        class_counts = y_class.value_counts()
        for class_name, count in class_counts.items():
            st.write(f"  - {class_name}: {count} samples")

        if len(X) < 10:
            st.error("❌ Insufficient samples for modeling (need at least 10 samples).")
            return None, None, None, None

        # Check if we have enough samples for each class
        min_class_size = class_counts.min()
        if min_class_size < 2:
            st.warning(
                f"⚠️ Some classes have very few samples (minimum: {min_class_size}). Consider collecting more data.")
            # Remove classes with only 1 sample
            valid_classes = class_counts[class_counts >= 2].index
            mask = y_class.isin(valid_classes)
            X = X[mask]
            y_class = y_class[mask]
            st.write(f"**After Removing Small Classes**: {len(X)} samples")

    except Exception as e:
        st.error(f"❌ Error in creating classification problem: {e}")
        return None, None, None, None

    # Split data with proper validation
    try:
        # Adjust test size if dataset is small
        test_size = min(0.2, max(0.1, len(X) * 0.2 / len(X)))  # Ensure reasonable split

        if len(X) < 20:
            test_size = 0.3  # Use larger test set for very small datasets

        # Check if stratification is possible
        unique_classes = y_class.value_counts()
        if unique_classes.min() >= 2:  # All classes have at least 2 samples
            stratify = y_class
        else:
            stratify = None
            st.warning("⚠️ Cannot stratify split due to class imbalance.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class,
            test_size=test_size,
            random_state=42,
            stratify=stratify
        )

        st.write(f"**Data Split**: Train: {len(X_train)}, Test: {len(X_test)}")

    except Exception as e:
        st.error(f"❌ Error in data splitting: {e}")
        return None, None, None, None

    # Train models with error handling
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),  # Reduced complexity
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5)
        # Reduced complexity
    }

    # Add SVM only if dataset is not too large
    if len(X_train) < 1000:
        models['SVM'] = SVC(random_state=42, probability=True)

    results = {}

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**{target_choice} Prediction Model Performance:**")
        performance_data = []

        for name, model in models.items():
            try:
                with st.spinner(f"Training {name}..."):
                    # Train model
                    model.fit(X_train, y_train)

                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }

                    performance_data.append({
                        'Model': name,
                        'Accuracy': f"{accuracy:.3f}",
                        'F1-Score': f"{f1:.3f}"
                    })

                    st.success(f"✅ {name} trained successfully")

            except Exception as e:
                st.warning(f"⚠️ Model {name} failed: {e}")
                continue

        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, hide_index=True)
        else:
            st.error("❌ No models trained successfully.")
            return None, None, available_features, target_choice

    with col2:
        # Feature importance for best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_model = results[best_model_name]['model']

            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': available_features,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)

                fig_imp = px.bar(importance_df, x='Importance', y='Feature',
                                 orientation='h', title=f'Top Features ({best_model_name})')
                fig_imp.update_layout(height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
            elif hasattr(best_model, 'coef_'):
                # For logistic regression, show coefficient magnitudes
                if len(best_model.coef_.shape) > 1:
                    coef_importance = np.abs(best_model.coef_).mean(axis=0)
                else:
                    coef_importance = np.abs(best_model.coef_[0])

                importance_df = pd.DataFrame({
                    'Feature': available_features,
                    'Importance': coef_importance
                }).sort_values('Importance', ascending=False).head(10)

                fig_imp = px.bar(importance_df, x='Importance', y='Feature',
                                 orientation='h', title=f'Feature Coefficients ({best_model_name})')
                fig_imp.update_layout(height=400)
                st.plotly_chart(fig_imp, use_container_width=True)

    # Confusion Matrix and detailed results
    if results:
        st.subheader("📈 Model Validation")
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_predictions = results[best_model_name]['predictions']

        # Confusion matrix
        cm = confusion_matrix(y_test, best_predictions)
        labels = best_model.classes_

        fig_cm = px.imshow(cm,
                           text_auto=True,
                           aspect="auto",
                           title=f'{best_model_name} - Confusion Matrix',
                           labels=dict(x="Predicted", y="Actual"),
                           x=labels,
                           y=labels)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        st.subheader("🎯 Detailed Classification Report")
        class_report = classification_report(y_test, best_predictions, output_dict=True)
        report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(report_df.round(3))

        return results, best_model_name, available_features, target_choice
    else:
        return None, None, available_features, target_choice


def clean_data(df):
    """Enhanced cleaning with comprehensive preprocessing"""
    metrics = {
        'initial_rows': len(df),
        'type_conversions': 0,
        'garage_replacements': {
            "Attched": "Attchd", "Basment": "Basement",
            "Attachd": "Attchd", "Detched": "Detchd"
        },
        'house_style_mapping': {
            '1.5Fin': '1.5_Story',
            '2.5Unf': '2.5_Story',
            'SFoyer': 'Split_Foyer'
        }
    }

    with st.status("🔧 Comprehensive Data Cleaning...", expanded=True) as status:
        # Remove duplicates
        df = df.drop_duplicates(subset="Id" if "Id" in df.columns else df.columns[0])
        metrics['duplicates_removed'] = metrics['initial_rows'] - len(df)

        # Enhanced numeric type handling
        integer_cols = ["YearBuilt", "YearRemodAdd", "OverallQual", "OverallCond",
                        "BedroomAbvGr", "KitchenAbvGr"]

        for col in integer_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'\D+', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Set appropriate defaults
                if col in ['OverallQual', 'OverallCond']:
                    col_median = 5
                elif col == 'YearBuilt':
                    col_median = 1980
                elif col == 'YearRemodAdd':
                    col_median = 1990
                else:
                    col_median = 1

                df[col] = df[col].fillna(col_median).astype(int)

        # Area column cleaning
        if 'Area' in df.columns:
            df["Area"] = (df["Area"]
                          .astype(str)
                          .str.replace(",", ".")
                          .str.replace(r'[^0-9.]', '', regex=True)
                          .replace('', np.nan))

            df["Area"] = pd.to_numeric(df["Area"], errors='coerce')
            df["Area"] = df["Area"].fillna(1500.0).clip(lower=0)

        # Categorical standardization
        if 'Garage' in df.columns:
            df["Garage"] = (df["Garage"]
                            .astype(str)
                            .replace(metrics['garage_replacements'])
                            .fillna("Unknown")
                            .replace('nan', 'Unknown'))

        if 'HouseStyle' in df.columns:
            df["HouseStyle"] = (df["HouseStyle"]
                                .astype(str)
                                .replace(metrics['house_style_mapping'])
                                .fillna("1Story")
                                .replace('nan', '1Story'))

        # Fill categorical NaN values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')

        # Temporal validation
        if all(col in df.columns for col in ["YearBuilt", "YearRemodAdd"]):
            df["YearRemodAdd"] = df[["YearBuilt", "YearRemodAdd"]].max(axis=1)

        # Final cleanup
        df = df.fillna(0)

        metrics.update({
            'remaining_rows': len(df),
            'final_missing': df.isna().sum().sum()
        })

        status.update(label="✅ Cleaning Successful!", state="complete", expanded=False)

    return df, metrics


def generate_quality_report(initial_stats, metrics, cleaned_df, exploration_results, model_results, best_model_name,
                            target_choice):
    """Generate comprehensive quality analysis report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#2E86AB')
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#A23B72')
    )

    # Title
    elements.append(Paragraph(f"Housing {target_choice} Prediction Analysis Report", title_style))
    elements.append(Spacer(1, 20))

    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    summary_text = f"""
    This comprehensive analysis of {initial_stats['total_rows']} housing properties focuses on predicting 
    {target_choice} ratings using property characteristics. The {best_model_name} model achieved the best 
    performance for quality assessment classification.
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 20))

    # Data Quality Section
    elements.append(Paragraph("Data Quality Assessment", heading_style))
    quality_data = [
        ['Metric', 'Value'],
        ['Initial Records', str(initial_stats['total_rows'])],
        ['Records After Cleaning', str(metrics['remaining_rows'])],
        ['Duplicates Removed', str(metrics['duplicates_removed'])],
        ['Final Missing Values', str(metrics.get('final_missing', 0))]
    ]

    quality_table = Table(quality_data)
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(quality_table)
    elements.append(Spacer(1, 20))

    # Key Findings
    elements.append(Paragraph("Key Quality Assessment Findings", heading_style))
    for i, insight in enumerate(exploration_results.get('key_insights', []), 1):
        elements.append(Paragraph(f"{i}. {insight}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # Model Performance
    elements.append(Paragraph(f"{target_choice} Prediction Model Performance", heading_style))

    if model_results:
        model_data = [['Model', 'Accuracy', 'F1-Score']]
        for name, results in model_results.items():
            model_data.append([
                name,
                f"{results['accuracy']:.3f}",
                f"{results['f1_score']:.3f}"
            ])

        model_table = Table(model_data)
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(model_table)

    elements.append(Spacer(1, 20))

    # Recommendations
    elements.append(Paragraph("Strategic Recommendations", heading_style))
    recommendations = [
        f"Use {target_choice} prediction models for property assessment automation",
        "Focus on key features identified through feature importance analysis",
        "Regular model retraining with new property assessment data",
        "Consider ensemble methods for improved prediction accuracy"
    ]

    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def main():
    """Main application function orchestrating the entire pipeline"""

    # Initialize session state
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'exploration_results' not in st.session_state:
        st.session_state.exploration_results = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None

    # Sidebar for navigation
    st.sidebar.title("🏠 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Step:",
        ["📁 Data Upload", "🔍 Data Exploration", "🤖 Model Training", "📊 Results & Download"]
    )

    # Main content based on selected page
    if page == "📁 Data Upload":
        st.header("📁 Data Upload & Preprocessing")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Excel file with housing data",
            type=['xlsx', 'xls'],
            help="Upload Excel file containing housing property data"
        )

        if uploaded_file is not None:
            try:
                # Load data
                raw_data = load_data(uploaded_file)
                st.success(f"✅ Data loaded successfully: {len(raw_data)} records")

                # Display initial statistics
                st.subheader("📈 Initial Data Overview")
                initial_stats = {
                    'total_rows': len(raw_data),
                    'total_columns': len(raw_data.columns),
                    'missing_values': raw_data.isna().sum().sum(),
                    'duplicate_rows': raw_data.duplicated().sum()
                }

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", initial_stats['total_rows'])
                with col2:
                    st.metric("Columns", initial_stats['total_columns'])
                with col3:
                    st.metric("Missing Values", initial_stats['missing_values'])
                with col4:
                    st.metric("Duplicates", initial_stats['duplicate_rows'])

                # Show sample data
                st.subheader("🔍 Sample Data Preview")
                st.dataframe(raw_data.head(10), use_container_width=True)

                # Data validation with schema
                st.subheader("✅ Data Validation")
                try:
                    # Validate against schema (will show warnings for missing columns)
                    available_schema_cols = [col for col in schema.columns.keys() if col in raw_data.columns]
                    if available_schema_cols:
                        partial_schema = pa.DataFrameSchema({
                            col: schema.columns[col] for col in available_schema_cols
                        })
                        validated_data = partial_schema.validate(raw_data[available_schema_cols])
                        st.success(f"✅ Data validation passed for {len(available_schema_cols)} columns")
                    else:
                        st.warning("⚠️ No matching schema columns found, proceeding with available data")
                        validated_data = raw_data
                except pa.errors.SchemaError as e:
                    st.warning(f"⚠️ Schema validation warnings: {str(e)[:200]}...")
                    validated_data = raw_data

                # Clean data
                if st.button("🔧 Clean Data", type="primary"):
                    cleaned_data, cleaning_metrics = clean_data(validated_data)

                    # Store in session state
                    st.session_state.cleaned_data = cleaned_data
                    st.session_state.initial_stats = initial_stats
                    st.session_state.cleaning_metrics = cleaning_metrics

                    # Show cleaning results
                    st.subheader("🎯 Cleaning Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Records Before", cleaning_metrics['initial_rows'])
                        st.metric("Records After", cleaning_metrics['remaining_rows'])

                    with col2:
                        st.metric("Duplicates Removed", cleaning_metrics['duplicates_removed'])
                        st.metric("Final Missing Values", cleaning_metrics.get('final_missing', 0))

                    st.success("✅ Data cleaning completed! Go to 'Data Exploration' to continue.")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your Excel file contains the expected columns for housing data.")

    elif page == "🔍 Data Exploration":
        st.header("🔍 Data Exploration & Analysis")

        if st.session_state.cleaned_data is not None:
            # Perform exploration
            exploration_results = explore_dataset(st.session_state.cleaned_data)
            st.session_state.exploration_results = exploration_results

            # Additional insights section
            st.subheader("📋 Summary of Key Insights")
            insights_df = pd.DataFrame({
                'Finding': [
                    'Quality-Condition Relationship',
                    'Age Impact on Quality',
                    'Area-Quality Correlation',
                    'Garage Impact',
                    'Room Configuration Effect'
                ],
                'Correlation/Impact': [
                    f"{exploration_results['quality_condition_correlation']:.3f}",
                    f"{exploration_results['age_quality_correlation']:.3f}",
                    f"{exploration_results['area_quality_correlation']:.3f}",
                    "Significant",
                    "Moderate"
                ],
                'Insight': [
                    "Quality and condition measure different aspects",
                    "Newer properties generally have higher quality",
                    "Larger properties tend to have higher quality",
                    "Built-in garages correlate with higher quality",
                    "Optimal bedroom count affects quality rating"
                ]
            })
            st.dataframe(insights_df, use_container_width=True, hide_index=True)

            if st.button("🚀 Proceed to Model Training", type="primary"):
                st.success("✅ Exploration completed! Go to 'Model Training' to build prediction models.")

        else:
            st.warning("⚠️ Please upload and clean data first in the 'Data Upload' section.")

    elif page == "🤖 Model Training":
        st.header("🤖 Model Training & Evaluation")

        if st.session_state.cleaned_data is not None:
            # Build models
            model_results, best_model_name, features_used, target_choice = build_quality_prediction_model(
                st.session_state.cleaned_data
            )

            # Store results
            st.session_state.model_results = model_results
            st.session_state.best_model_name = best_model_name
            st.session_state.features_used = features_used
            st.session_state.target_choice = target_choice
            st.session_state.analysis_completed = True

            if model_results:
                st.success(f"✅ Model training completed! Best model: {best_model_name}")

                # Model comparison
                st.subheader("🏆 Model Performance Comparison")
                comparison_data = []
                for name, results in model_results.items():
                    comparison_data.append({
                        'Model': name,
                        'Accuracy': results['accuracy'],
                        'F1-Score': results['f1_score'],
                        'Status': '🥇 Best' if name == best_model_name else '✅ Good'
                    })

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                # Feature importance summary
                if best_model_name and st.session_state.model_results:
                    best_model = st.session_state.model_results[best_model_name]['model']

                    st.subheader("🎯 Most Important Features")
                    if hasattr(best_model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': features_used,
                            'Importance': best_model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(5)

                        for idx, row in importance_df.iterrows():
                            st.write(f"**{row['Feature']}**: {row['Importance']:.3f}")

                if st.button("📊 View Results & Generate Report", type="primary"):
                    st.success("✅ Training completed! Go to 'Results & Download' for the final report.")

        else:
            st.warning("⚠️ Please complete data upload and exploration first.")

    elif page == "📊 Results & Download":
        st.header("📊 Analysis Results & Report Generation")

        if st.session_state.analysis_completed and st.session_state.model_results:
            # Final results summary
            st.subheader("🎉 Analysis Complete!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Model", st.session_state.best_model_name)
            with col2:
                best_accuracy = st.session_state.model_results[st.session_state.best_model_name]['accuracy']
                st.metric("Best Accuracy", f"{best_accuracy:.3f}")
            with col3:
                st.metric("Target Predicted", st.session_state.target_choice)

            # Key recommendations
            st.subheader("💡 Key Recommendations")
            recommendations = [
                f"Deploy {st.session_state.best_model_name} for {st.session_state.target_choice} prediction",
                "Focus on top-performing features for property assessment",
                "Regular model retraining with new data recommended",
                "Consider ensemble methods for production deployment"
            ]

            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

            # Generate and download report
            st.subheader("📥 Download Analysis Report")

            if st.button("📄 Generate PDF Report", type="primary"):
                try:
                    report_buffer = generate_quality_report(
                        st.session_state.initial_stats,
                        st.session_state.cleaning_metrics,
                        st.session_state.cleaned_data,
                        st.session_state.exploration_results,
                        st.session_state.model_results,
                        st.session_state.best_model_name,
                        st.session_state.target_choice
                    )

                    st.download_button(
                        label="📥 Download PDF Report",
                        data=report_buffer,
                        file_name=f"housing_{st.session_state.target_choice.lower()}_analysis_report.pdf",
                        mime="application/pdf"
                    )

                    st.success("✅ Report generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")

            # Option to restart analysis
            if st.button("🔄 Start New Analysis"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        else:
            st.warning("⚠️ Please complete the full analysis pipeline first.")
            st.info("Steps: Data Upload → Data Exploration → Model Training → Results")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Analysis Pipeline")
    st.sidebar.markdown("1. **Upload**: Load Excel data")
    st.sidebar.markdown("2. **Explore**: Analyze patterns")
    st.sidebar.markdown("3. **Model**: Train predictors")
    st.sidebar.markdown("4. **Report**: Generate results")

    # Progress indicator
    progress_steps = {
        "📁 Data Upload": st.session_state.cleaned_data is not None,
        "🔍 Data Exploration": st.session_state.exploration_results is not None,
        "🤖 Model Training": st.session_state.model_results is not None,
        "📊 Results & Download": st.session_state.analysis_completed
    }

    st.sidebar.markdown("### 🎯 Progress")
    for step, completed in progress_steps.items():
        status = "✅" if completed else "⏳"
        st.sidebar.markdown(f"{status} {step.split(' ', 1)[1]}")


# Run the application
if __name__ == "__main__":
    main()