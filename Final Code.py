# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:48:52 2024

@author: barab
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
train = pd.read_csv("C:/Users/barab/OneDrive/Documents/McGill MMA/Courses/MGSC 673/train.csv")

# Define all numeric and categorical variables
numeric_vars = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
                'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
                'MoSold', 'YrSold']

categorical_vars = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
                    'Condition1', 'Condition2', 'RoofStyle',
                    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
                    'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
                    'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
                    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
                    'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Fill missing values
train[categorical_vars] = train[categorical_vars].fillna('None')
train[numeric_vars] = train[numeric_vars].fillna(train[numeric_vars].median())

# Define function to create house categories
def create_house_category(df):
    conditions = [
        # Historic vs. Modern Homes
        ((df['YearBuilt'] < 1950) & (df['YearRemodAdd'] < 1950)),
        ((df['YearBuilt'] >= 1950) & (df['YearRemodAdd'] >= 1950)),
        # Home Size (based on number of floors)
        (df['HouseStyle'].isin(['2Story', '2.5Fin', '2.5Unf'])),
        (df['HouseStyle'].isin(['1Story', '1.5Fin', '1.5Unf'])),
        # Architectural Style
        (df['BldgType'] == '1Fam') & (df['HouseStyle'].isin(['1Story', '1.5Fin'])),
        (df['BldgType'] == '1Fam') & (df['HouseStyle'].isin(['2Story', '2.5Fin'])),
        # Neighborhood Characteristics
        ((df['YearBuilt'] < 1950) & (df['BldgType'] == '1Fam')),
        ((df['YearBuilt'] >= 1950) & (df['BldgType'] == '1Fam')),
    ]
    choices = ['Historic', 'Modern', 'Two-Story', 'One-Story', 'Single-Family One-Story', 'Single-Family Two-Story', 'Old Neighborhood', 'New Neighborhood']
    df['HouseCategory'] = np.select(conditions, choices, default='Other')
    return df

# Apply the house category creation
train = create_house_category(train)

# Drop unnecessary columns to prevent data leakage
train.drop(['YearBuilt', 'YearRemodAdd', 'BldgType', 'HouseStyle'], axis=1, inplace=True)

# Define preprocessing pipeline
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
from sklearn.preprocessing import RobustScaler

# Update the numeric pipeline in the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', numeric_imputer), ('scaler', RobustScaler())]), numeric_vars),
        ('cat', Pipeline([('imputer', categorical_imputer), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_vars)
    ])


# Encode category labels numerically
label_encoder = LabelEncoder()
train['HouseCategory'] = label_encoder.fit_transform(train['HouseCategory'])

# Split the data into training and testing subsets
train_set, test_set = train_test_split(train, test_size=0.2, random_state=42)

# Preprocess the data
X_train = preprocessor.fit_transform(train_set[numeric_vars + categorical_vars])
X_test = preprocessor.transform(test_set[numeric_vars + categorical_vars])

# Convert sparse matrix to dense if necessary
X_train = X_train.toarray() if sparse.issparse(X_train) else X_train
X_test = X_test.toarray() if sparse.issparse(X_test) else X_test

y_train_price = train_set['SalePrice']
y_test_price = test_set['SalePrice']
y_train_category = train_set['HouseCategory']
y_test_category = test_set['HouseCategory']

import torch.nn.functional as F

class MultiTaskModel(pl.LightningModule):
    def __init__(self, input_dim, num_classes, lr=1e-3, hidden_size=128, dropout_rate=0.1):
        super().__init__()
        self.save_hyperparameters()
        # Increasing complexity by adding more layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_size * 2),  # Increase initial layer size
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size * 2),  # Increase intermediate layer size
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )         

        self.regression_head = nn.Linear(64, 1)
        self.classification_head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.shared(x)
        price = self.regression_head(x).view(-1, 1)
        category = self.classification_head(x)
        return price, category

    def training_step(self, batch, batch_idx):
        x, y_price, y_cat = batch
        pred_price, pred_cat = self(x)
        
        # Define custom weights for regression and classification losses
        weight_price = 0.5  # Weight for regression loss
        weight_category = 0.5  # Weight for classification loss
        
        # Calculate weighted loss
        loss = (weight_price * F.mse_loss(pred_price, y_price)) + (weight_category * F.cross_entropy(pred_cat, y_cat))
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        scheduler_config = {
            'scheduler': lr_scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'train_loss'
        }
        return [optimizer], [scheduler_config]

# Setup logging and callbacks
logger = TensorBoardLogger("tb_logs", name="my_model")
checkpoint_callback = ModelCheckpoint(monitor='train_loss', dirpath='model/', filename='model-{epoch:02d}-{train_loss:.2f}', save_top_k=1, mode='min')

# Convert pandas dataframes to tensors
X_train_tensor = torch.tensor(X_train.astype(np.float32))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_train_price_tensor = torch.tensor(y_train_price.values.astype(np.float32)).view(-1, 1)
y_test_price_tensor = torch.tensor(y_test_price.values.astype(np.float32)).view(-1, 1)
y_train_category_tensor = torch.tensor(y_train_category.values).long()
y_test_category_tensor = torch.tensor(y_test_category.values).long()

# Create dataset and dataloader for training
train_dataset = TensorDataset(X_train_tensor, y_train_price_tensor, y_train_category_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create dataset and dataloader for testing
test_dataset = TensorDataset(X_test_tensor, y_test_price_tensor, y_test_category_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class OptunaCallback(Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_train_end(self, trainer, pl_module):
        self.trial.report(trainer.callback_metrics['train_loss'].item(), step=trainer.global_step)
        if trainer.should_stop:
            self.trial.report(trainer.callback_metrics['train_loss'].item(), step=trainer.global_step)
            raise optuna.exceptions.TrialPruned()

def objective(trial):
    # Define hyperparameters to search
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 30, 500)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0001, 0.9)
    lr = trial.suggest_float('lr', 1e-4, 0.1, log=True)
    
    # Define Lightning Module
    class MultiTaskModel(pl.LightningModule):
        def __init__(self, input_dim, num_classes, lr, hidden_size, dropout_rate):
            super().__init__()
            self.save_hyperparameters()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            self.regression_head = nn.Linear(hidden_size, 1)
            self.classification_head = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.shared(x)
            price = self.regression_head(x).view(-1, 1)
            category = self.classification_head(x)
            return price, category

        def training_step(self, batch, batch_idx):
            x, y_price, y_cat = batch
            pred_price, pred_cat = self(x)
            loss = F.mse_loss(pred_price, y_price) + F.cross_entropy(pred_cat, y_cat)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
            return optimizer
    
    # Create model instance
    model = MultiTaskModel(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train_category)),
                           lr=lr, hidden_size=hidden_size, dropout_rate=dropout_rate)

    # Setup logging and callbacks
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer(max_epochs=100, logger=logger, callbacks=[checkpoint_callback, OptunaCallback(trial)])
    
    # Train the model
    trainer.fit(model, train_loader)
    
    # Return the loss to Optuna
    return trainer.callback_metrics['train_loss'].item()

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters found
print("Best hyperparameters:", study.best_params)

def evaluate_model(model, test_loader):
    model.eval()
    predicted_categories = []
    true_categories = []
    predicted_prices = []
    true_prices = []

    with torch.no_grad():
        for batch in test_loader:
            X, y_price, y_category = batch
            predicted_price, predicted_cat = model(X)
            predicted_prices.extend(predicted_price.view(-1).cpu().numpy())  # Flatten to match y_price
            true_prices.extend(y_price.view(-1).cpu().numpy())  # Flatten to match predicted_price
            predicted_categories.extend(torch.argmax(predicted_cat, dim=1).cpu().numpy())
            true_categories.extend(y_category.cpu().numpy())

    def calculate_regression_metrics(true_prices, predicted_prices):
        mse = mean_squared_error(true_prices, predicted_prices)
        rmse = mean_squared_error(true_prices, predicted_prices, squared=False)
        mae = mean_absolute_error(true_prices, predicted_prices)
        r2 = r2_score(true_prices, predicted_prices)
        return mse, rmse, mae, r2

    def calculate_classification_metrics(true_categories, predicted_categories):
        accuracy = accuracy_score(true_categories, predicted_categories)
        precision = precision_score(true_categories, predicted_categories, average='weighted')
        recall = recall_score(true_categories, predicted_categories, average='weighted')
        f1 = f1_score(true_categories, predicted_categories, average='weighted')
        return accuracy, precision, recall, f1

    # Calculate regression metrics
    regression_metrics = calculate_regression_metrics(true_prices, predicted_prices)

    # Calculate classification metrics
    classification_metrics = calculate_classification_metrics(true_categories, predicted_categories)

    return regression_metrics, classification_metrics

import optuna

# Define the objective function for Optuna to minimize
def objective(trial):
    # Define hyperparameters to be tuned
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    # Instantiate the model with suggested hyperparameters
    model = MultiTaskModel(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train_category)),
                           lr=lr, hidden_size=hidden_size, dropout_rate=dropout_rate)

    # Define PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=100)

    # Train the model
    trainer.fit(model, train_loader)

    # Evaluate the model
    regression_metrics, classification_metrics = evaluate_model(model, test_loader)

    # Return the loss to be minimized (e.g., MSE for regression)
    return regression_metrics[0]  # Returning MSE for minimization

# Perform hyperparameter optimization with Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Instantiate the model with the best hyperparameters
best_model = MultiTaskModel(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train_category)),
                            lr=best_params["lr"], hidden_size=best_params["hidden_size"],
                            dropout_rate=best_params["dropout_rate"])

# Train the best model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(best_model, train_loader)

# Evaluate the best model
regression_metrics, classification_metrics = evaluate_model(best_model, test_loader)

# Print the evaluation metrics
print("Regression Metrics:")
print(f"MSE: {regression_metrics[0]}")
print(f"RMSE: {regression_metrics[1]}")
print(f"MAE: {regression_metrics[2]}")
print(f"R2 Score: {regression_metrics[3]}")

print("\nClassification Metrics:")
print(f"Accuracy: {classification_metrics[0]}")
print(f"Precision: {classification_metrics[1]}")
print(f"Recall: {classification_metrics[2]}")
print(f"F1-score: {classification_metrics[3]}")

torch.save(best_model.state_dict(), "best_model.pth")
