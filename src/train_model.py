from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle

from src.config import *
from src.evaluate_model import evaluate_model

def train_model(X_train,X_test,y_train,y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(),
        'Ridge Regression': Ridge(),
        'Elastic Net Regression':ElasticNet(),
        'AdaBoost': AdaBoostRegressor(),
        'KNN': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor()
    }
    model_results = {
        'Model_Name': [],
        'Train_RMSE': [], 'Train_MAE': [], 'Train_R2': [], 'Train_Adj_R2': [],
        'Test_RMSE': [], 'Test_MAE': [], 'Test_R2': [], 'Test_Adj_R2': [],
        'Overfitting_Gap': [] 
    }
    for i in range(len(list(models))):
        model_name = list(models.keys())[i]
        model = list(models.values())[i]
        
        # Train the model
        model.fit(X_train, y_train) 
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate the model
        mae_train, rmse_train, r2_train, adj_r2_train = evaluate_model(X_train,y_train, y_train_pred)
        mae_test, rmse_test, r2_test, adj_r2_test = evaluate_model(X_test,y_test, y_test_pred)

        model_results['Model_Name'].append(model_name)
        
        # Storing training Performance Metrics
        model_results['Train_RMSE'].append(rmse_train)
        model_results['Train_MAE'].append(mae_train)
        model_results['Train_R2'].append(r2_train)
        model_results['Train_Adj_R2'].append(adj_r2_train)    
        
        # Storing testing Performance Metrics
        model_results['Test_RMSE'].append(rmse_test)
        model_results['Test_MAE'].append(mae_test)
        model_results['Test_R2'].append(r2_test)
        model_results['Test_Adj_R2'].append(adj_r2_test)
        model_results['Overfitting_Gap'].append(round(r2_train-r2_test,2))

    #In this cell, we perform Hyperparameter Tuning for optimal resutls. We will do the tuning for two best models.

    params_ada ={
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.5, 1.0,10],
        'loss': ['linear', 'square', 'exponential']
    }

    params_RF ={
        'n_estimators': [100, 300, 500],
        'max_features': [0.6, 0.8, 1.0, 'sqrt'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
    }

    tune_models =[
        ('AdaBoost',AdaBoostRegressor(),params_ada),
        ('RandomForest',RandomForestRegressor(),params_RF)
    ]
    from sklearn.model_selection import RandomizedSearchCV

    model_params = {}

    for name,model,params in tune_models:
        random = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=50,
            cv=5,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )
        random.fit(X_train,y_train)
        model_params[name] = random.best_params_

    # Here we will be seeing how much the performance has increased by using the tuned parameters for each of the tuned models.
    model_optimized = {
        'RandomForest': RandomForestRegressor(n_estimators=100,min_samples_split=5,min_samples_leaf=1,max_features=0.6,max_depth=None),
        'AdaBoost': AdaBoostRegressor(n_estimators=50,loss='linear',learning_rate=1)
    }

    for i in range(len(list(model_optimized))):
        model_name = list(model_optimized.keys())[i]
        model = list(model_optimized.values())[i]
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        mae_train, rmse_train, r2_train, adj_r2_train = evaluate_model(X_train,y_train, y_train_pred)
        mae_test, rmse_test, r2_test, adj_r2_test = evaluate_model(X_test,y_test, y_test_pred)
    
    best_model = model_optimized['RandomForest']
    pickle_file = 'model/RF_regressor.pkl'    
    with open (pickle_file, 'wb') as file:
        pickle.dump(best_model, file)
    print(f'Model saved as {pickle_file}')
    

        

