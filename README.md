# predictive_modeling_credit_risk
In the following repository, the use of two custom classes are demonstrated on the following loan data: [link](https://github.com/Humboldt-WI/bads/blob/master/data/loan_data.csv):
- CategoryOptimizer - A custom class that extends upon an optimization function using $\chi^2$ testing for merging category levels demonstrated in class by Prof. Dr. Stefan Lessmann at Hu Berlin. The class is demonstrated in the following [notebook](https://github.com/anasashb/predictive_modeling_credit_risk/blob/main/data_prep_and_selection/prep_notebook.ipynb) of this repository.
- ModelValidator - A custom class that serves as an easy-to-implement 5-fold cross validation for Logistic Regression, Random Forest Classifier and XGBoost Classifier. Class includes methods for validating the three models, as well as conducting grid search on RF and XGB. The class is demonstrated in the following [notebook](https://github.com/anasashb/predictive_modeling_credit_risk/blob/main/model_validation/val_notebook.ipynb) of this repository. 
- - -
``` python
# Class to optimize grouping using X^2 test

class CategoryOptimizer:

    def __init__(self, categorical_feature, target_feature):
        self.categorical_feature = categorical_feature
        self.target_feature = target_feature

        self.category_amount = [self.categorical_feature.nunique()]
        self.categories = [self.categorical_feature.cat.categories]
        # Some empty containers as callable self arguments
        self.test_statistics = []
        self.p_values = []
        # Optimize method automatically
        self._optimize()

    
    def _optimize(self):
        '''
        Computes crosstab of the categorical feature and target feature. 
        Computes Good/Bad odds ratio differentials and optimizes grouping based on minimum differences in ratio.
        Best grouping selected using X^2 test.
        '''

        # First do the X^2 on unmerged data and append to containers
        cross_tab = pd.crosstab(self.categorical_feature, self.target_feature)
        stat, p_val, _, _ = stats.chi2_contingency(cross_tab)
        self.test_statistics.append(stat)
        self.p_values.append(p_val)

        # Begin iterative grouping
        while self.category_amount[-1] > 1:
            cross_tab = pd.crosstab(self.categorical_feature, self.target_feature)
            # Get odds ratio
            cross_tab['odds'] = cross_tab[0] / cross_tab[1]
            # Sort
            cross_tab.sort_values('odds', inplace=True)
            # Calculate differences in odds between neighboring categories
            cross_tab['diff'] = cross_tab['odds'].diff()
            # Find where difference in odds minimum
            minimum_index = np.where(cross_tab['diff']==cross_tab['diff'].min())[0][0]
            # Identify levels to merge
            levels_to_merge = cross_tab[(minimum_index-1):(minimum_index+1)].index.values
            # Generate New Level Name
            new_level = '+'.join(levels_to_merge)
            # Add New Level as Category
            self.categorical_feature = self.categorical_feature.cat.add_categories(new_level)
            # Assign Data to New Level
            for l in levels_to_merge:
                self.categorical_feature[self.categorical_feature == l] = new_level
            # Remove old levels
            self.categorical_feature = self.categorical_feature.cat.remove_categories(levels_to_merge)
            # Append to category amount and categories lists
            self.category_amount.append(self.categorical_feature.nunique())
            self.categories.append(self.categorical_feature.cat.categories)
            #Chi^2 for Merged Category and append to containers
            cross_tab = pd.crosstab(self.categorical_feature, self.target_feature)
            stat, p_val, _, _ = stats.chi2_contingency(cross_tab)
            self.test_statistics.append(stat)
            self.p_values.append(p_val)
    
    def elbow_plot(self):
        '''
        Makes an elbow plot for the chi^2 tests conducted per iteration
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('Elbow Cruve for $\chi^2$')
        ax1.plot(self.category_amount, self.test_statistics, '#1f77b4')
        ax1.set_xlabel('No. of Categories')
        ax1.set_ylabel('$\chi^2$ test statistic', color='#1f77b4' )

        ax2 = ax1.twinx()
        ax2.plot(self.category_amount, self.p_values, '#d62728')
        ax2.set_ylabel('$\chi^2$ p-value', color='#d62728')
        ax2.set_ylim(0, 1e-7)

        plt.show()

    
    def print_results(self):
        '''
        Prints all considered merging options and corresponding chi^2 p-values
        '''
        for i in range(len(self.categories)):
            print(f'{self.category_amount[i]} Categories: {self.categories[i].values}\n'
                  f'Chi-square p-value: {self.p_values[i]}')
            print('-'*80)


    def get_best_grouping(self):
        '''
        Prints only the best grouping option
        '''
        minimum_p_val_index = self.p_values.index(min(self.p_values))

        print(f'{self.category_amount[minimum_p_val_index]} Categories: {self.categories[minimum_p_val_index].values}\n'
              f'Chi-square p-value: {self.p_values[minimum_p_val_index]}')
```
- - -
```python
# Crossvalidation class
class ModelValidator:
    '''
    A custom class to wrap up model k-fold validation functionalities.
    '''

    # Defining initial arguments
    def __init__(self, X, y):
        # Defining initial class inputs as self arguments
        self.X = X
        self.y = y

    def Logit_validate(self):
        '''
        Fits a Logistic regression model given predictors and a target. 
        
        Args:
            alpha: The L2 penalty value - also called lambda for Ridge Regression.
            tuning: Boolean argument. If set to True, a grid search will be carried out for alpha values
                between 0.01 and 1 with step size of 0.01. 
                Note that setting to 0 (linear regression without any penalties ) will cause model to break down.
        Returns:
            logit_scores: An array of 15 F1 scores obtained from fitting the five folds three times. 
        '''

        # Define cross validation method
        # Use five fold
        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = 66)
        # Set up model
        model = LogisticRegression()

        # To avoid scaling dummy variables
        non_binary = [col for col in self.X.columns if self.X[col].nunique()>2]
        scaler = ColumnTransformer(
            transformers=[
                ('scale', StandardScaler(), non_binary)
                ],
                remainder='passthrough'
                )  

        # Using make_pipeline for our purposes should ensure there is no data leakage 
        # while scaling the folds
        pipeline = make_pipeline(scaler, model)
        
        # F1 for evaluation, as we don't want low recall on positive class
        logit_scores = cross_val_score(pipeline, self.X, self.y, scoring = 'f1', cv = cv, n_jobs = -1)

        print('Results:')
        print('-'*100)
        print(f'Logistic Regression F1: {np.mean(logit_scores):.4f} | Standard Deviation: {np.std(logit_scores):.4f} |')
        print('='*100)

        return logit_scores
    
    
    def RF_search(self,
                  estimator_range=(50,1001,50),
                  depth_range=(10,51,10)):

        '''
        Simple method to conduct grid search for Random Forest Classifier.
        
        Args:
            estimator_range: (min, max, step) tuple of integers for n_estimators in parameter grid 
            depth_range: (min, max, step) tuple of integers for max_depth in parameter grid 
        Returns:
            best_estimator, best_score
        '''
        # Set up cv
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=66)
        # Set up model 
        model = RandomForestClassifier(random_state=66)
        # Set up grid
        param_grid = {
            'n_estimators': list(range(estimator_range[0], estimator_range[1], estimator_range[2])),
            'max_depth': list(range(depth_range[0], depth_range[1], depth_range[2]))
        }
        # Set up search
        search = GridSearchCV(model, param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
        # Run search
        search.fit(self.X, self.y)
        print('Results:')
        print('-'*100)
        print(f'Best RF F1 score: {search.best_score_:.4f}')
        print(f'Best Parameters: {search.best_params_}')
        print('='*100)
        
        return search.best_estimator_, search.best_score_

    def RF_validate(self,
                    n_estimators = 100,
                    max_depth = 20,
                    min_samples_split = 2,
                    max_features = 'sqrt'):
        '''
        Fits and cross validates Random Forest classification model given predictors and a target.

        Args:
            n_estimators: How many trees to include in the ensemble (100 by default)
            max_depth: Maximum depth of a tree (20 by default)
            min_samples_split: Minimum amount of samples to split a node (2 by default)
            max_features: Number of features to consider when searching for best split ('sqrt' by default)

        Returns:
            rf_scores: An array of 15 F1 scores obtained from fitting the five folds three times.

        '''
        # Set up CV
        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = 66)
        model = RandomForestClassifier(n_estimators = n_estimators,
                                      max_depth = max_depth,
                                      min_samples_split = min_samples_split,
                                      max_features = max_features,
                                      random_state = 66
                                      )
        # Same scoring
        rf_scores = cross_val_score(model, self.X, self.y, scoring = 'f1', cv = cv, n_jobs = -1)
        
        print('Results:')
        print('-'*100)
        print(f'RF Classifier F1: {np.mean(rf_scores):.4f} | Standard Deviation: {np.std(rf_scores):.4f} |')
        print('='*100)

        return rf_scores
    

    def XGB_search(self,
                  estimator_range=(50,1001,50),
                  depth_range=(2,11,1),
                  lr_range=[0.01, 0.05, 0.1, 0.3, 0.5]):

        '''
        Simple method to conduct grid search for XGBoost Classifier.
        
        Args:
            estimator_range: (min, max, step) tuple of integers for n_estimators in parameter grid 
            depth_range: (min, max, step) tuple of integers for max_depth in parameter grid
            lr_range: list of floats for learning_rate in param grid 
        Returns:
            best_estimator, best_score
        '''
        # Set up cv
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=66)
        # Set up model 
        model = XGBClassifier(eval_metric='logloss', random_state=66)
        # Set up grid
        param_grid = {
            'n_estimators': list(range(estimator_range[0], estimator_range[1], estimator_range[2])),
            'max_depth': list(range(depth_range[0], depth_range[1], depth_range[2])),
            'learning_rate': lr_range
        }
        # Set up search
        search = GridSearchCV(model, param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=1)
        # Run search
        search.fit(self.X, self.y)
        print('Results:')
        print('-'*100)
        print(f'Best XGB F1 score: {search.best_score_:.4f}')
        print(f'Best Parameters: {search.best_params_}')
        print('='*100)
        
        return search.best_estimator_, search.best_score_        
    
    def XGB_validate(self,
                    n_estimators = 500,
                    max_depth = 5,
                    learning_rate=0.3):
        '''
        Fits an extreme gradient boosting classification model given predictors and a target.

        Args:
            n_estimators: How many trees to include in the ensemble (500 by default)
            max_depth: Maximum depth of a tree (5 by default)
            learning_rate: Learning rate (0.3 by default)

        Returns:
            xgb_scores: An array of 15 F1 scores obtained from fitting the five folds three times.

        '''
        # Set up CV
        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = 66)
        # Set up model
        model = XGBClassifier(n_estimators = n_estimators,
                                      max_depth = max_depth,
                                      learning_rate=learning_rate,
                                      random_state = 66
                                      )
        # Same pipeline
        # Same scoring
        xgb_scores = cross_val_score(model, self.X, self.y, scoring = 'f1', cv = cv, n_jobs = -1)
        
        print('Results:')
        print('-'*100)
        print(f'XGB Classifier F1: {np.mean(xgb_scores):.4f} | Standard Deviation: {np.std(xgb_scores):.4f} |')
        print('='*100)

        return xgb_scores

```
