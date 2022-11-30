# Librerías comunes utilizadas
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.base import BaseEstimator,TransformerMixin

    
class Fitter:
    
    # Default feature matrix
    features = pd.read_csv('data/fmatrix-base.csv', index_col=0)
    
    # Available MLC keys and full names
    available_models = {
        'xg': 'XGBoost', 
        'rf': 'Random Forest', 
        'lr': 'Logistic Regression', 
        'nb': 'Naive Bayes'
    }
    
    # Available CUS keys and full names
    available_scores = {
        'sofa': 'SOFA', 
        'oasis': 'OASIS', 
        'mods': 'MODS', 
        'apache': 'APACHE II', 
        'saps': 'SAPS II'
    }
    
    available_estimators = {**available_models, **available_scores}
    
    available_datasets = {
        'train': 'Training data',
        'test': 'Testing data',
        'val': 'Validation data'
    }
    
    
    def __init__(self, records=None, name='default', fmatrix='base'):
        
        self.fmatrix = pd.read_csv(f'data/fmatrix-{fmatrix}.csv', index_col=0)

        if records is None:
            self.records = pd.DataFrame(columns=self.fmatrix.index)
            self.records.index.name = 'idi'
        
        else:
            self.records = records
        
        if name is None:
            import datetime
            name = 'D' + datetime.datetime.now().strftime("%y%m%d%H%I%s")
        
        self.name = name
    
        
    def __str__(self):
        strrep = []
        
        if hasattr(self, 'name'):
            strrep.append(f'Fitter name: {self.name}')
        
        if hasattr(self, 'records'):
            strrep.append(f'Total dataset size: {len(self.records):,}')
            
        if hasattr(self, 'itrain'):
            strrep.append(f'Training dataset size: {len(self.itrain):,}')
        
        if hasattr(self, 'itest'):
            strrep.append(f'Testing dataset size: {len(self.itest):,}')
        
        if hasattr(self, 'ival'):
            strrep.append(f'Validation dataset size: {len(self.ival):,}')
    
        if hasattr(self, 'feature_names'):
            nfeatures = len(self.feature_names)
            nparams = len(self.features.loc[self.feature_names, 'parameter'].unique())
            strrep.append(f'Parameters used: {nparams}')
            strrep.append(f'Features used: {nfeatures}')
            
        return "\n".join(strrep)
        
        
    def _getX(self, estimator, dataset='train'):
        '''Get the appropiate X transformation for a given estimator.
        
        Parameters
        ----------
        estimator : str
            Key of the estimator. Must be in fitter.available_models.keys()
        dataset : str
            Key of the dataset. Must be in fitter.available_datasets.keys()
        
        Return
        ------
        numpy.array
            Transformed X dataset.
        '''
        
        if dataset == 'train':
            X = self.itrain
            
        elif dataset == 'test':
            X = self.itest
            
        elif dataset == 'val':
            X = self.ival
        
        if estimator in ['lr', 'svm']:
            return self.pipeline['quantile'].transform(X)
            
        elif estimator in self.available_models.keys():
            return self.pipeline['selector'].transform(X)
        
        else:
            return X
        
        
    def _crossvalidator(self, classifier, parameters, kfold):
        '''Makes a grid search cross-validation (sklearn.model_selection.GridSearchCV)
        to a classifier.
        
        Parameters
        ----------
        classifier : xgboost or sklearn estimator object
            This is assumed to implement the scikit-learn estimator interface. Either 
            estimator needs to provide a score function, or scoring must be passed.
        parameters : dict or list of dictionaries
            Dictionary with parameters names (str) as keys and lists of parameter 
            settings to try as values, or a list of such dictionaries, in which case 
            the grids spanned by each dictionary in the list are explored. This enables 
            searching over any sequence of parameter settings.
        kfold : int
            Number of folds for the Stratified K-Folds cross-validator. Must be at 
            least 2.
        
        Return
        ------
        sklearn.model_selection.GridSearchCV object
            Cross-validation object.
        '''
        
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import StratifiedKFold
        
        return GridSearchCV(
            classifier,
            parameters,
            n_jobs=-1,
            cv=StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0),
            scoring='average_precision',
            refit='average_precision'
        )
    

    def _confidence_interval(self, ytrue, ypred, metric, n_bootstrap=1000, alpha=0.05):
        '''Calculate confidence intervals by bootstrapping.
        
        Parameters
        ----------
        ytrue : array of shape (n_samples,)
            True targets.
        ypred : array of shape (n_samples,)
            Probabilities of the positive class.
        metric : str
            Metric used. Use 'ap' for average precision, any other value for
            ROC-AUC.
        n_bootstrap : int, default 1000
            Number of bootstrap iterations.
        alpha : float, default 0.05
            Confidence interval threshold. Default is 0.05, which leads to 90%
            confidence interval.
        
        Return
        ------
        tuple : ci_lower, ci_upper
            Tuple of confidence interval limits.
        '''
        
        bootstrapped_scores = []

        rng = np.random.RandomState(0)
        for i in range(n_bootstrap):

            # Bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(ypred), len(ypred))

            # We need at least one positive and one negative sample for AUC-ROC
            # to be defined. Reject the sample if this does not happen.
            if len(np.unique(ytrue[indices])) < 2: continue

            if metric == 'ap':
                score = sklearn.metrics.average_precision_score(ytrue[indices], ypred[indices])
            else:
                score = sklearn.metrics.roc_auc_score(ytrue[indices], ypred[indices])
                
            bootstrapped_scores.append(score)

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        ci_lower = sorted_scores[int(alpha * len(sorted_scores))]
        ci_upper = sorted_scores[int((1 - alpha) * len(sorted_scores))]
        
        return ci_lower, ci_upper
    
    
    def _set_background(self):
        '''Draw Background dataset from training dataset. First, estimate the sample 
        size required to create a representative sample (background dataset) from the 
        population (training dataset).
        
        Return
        ------
        pandas.DataFrame
            Background dataset.
        '''
        
        import math
        from numpy.random import default_rng
        import scipy.stats as stats
        
        # 1. Population proportion: Describes the proportion of a value associated with one feature in
        # the training dataset. For binary variables, maximum variance of the proportion must be 
        # assumed (0.5) because of some features approching this value. For continuous variables, there
        # is an qual chance of the sample mean being either higher or lower than the population mean. 
        # Hence, the population proportion remains of 0.5.
        p = 0.50
        
        # 2. Margin of error: Deviation of the sample mean from the population mean.
        e = 0.02
        
        # 3. Z-Score for 99% confidence level: This is the probability that the population mean falls 
        # within the sample mean +- the margin of error.
        z = 2.58
        
        # 4. Get population size
        population = self._getX('xg', 'train')
        popoutcome = self.ytrain
        N = len(population)
        
        # 5. Calculate needed sample size
        n = math.ceil((z**2 * p * (1 - p) / e**2) / (1 + ((z**2 * p * (1 - p)) / (e**2 * N))))

        # Create the background dataset. Stratify the sampling to mantain the mortality of the dataset
        rng = default_rng(0)
        bkgindices = rng.choice(N, n, replace=False)
        bkgoutcome = popoutcome.iloc[bkgindices]
        background = population[bkgindices, :]
        
        # Check that the sample is representative
        true_mort = popoutcome.mean()
        samp_mort = bkgoutcome.mean()
        mort_error = samp_mort * (1 - e), samp_mort * (1 + e)
        mort_ttest = stats.ttest_1samp(bkgoutcome, true_mort).pvalue

        if mort_ttest < 0.05:
            print(f'Sample mortality ttest: {mort_ttest:.4f}')
            print(f'Sample mortality true value [margin of error]: {true_mort:.2f} [{mort_error[0]:.2f} - {mort_error[1]:.2f}]')

        for i in range(background.shape[1]):
            true_mean = population[:, i].mean()
            samp_mean = background[:, i].mean()
            samp_error = samp_mean * (1 - e), samp_mean * (1 + e)
            samp_ttest = stats.ttest_1samp(background[:, i], true_mean).pvalue

            if samp_ttest < 0.05 and true_mean - samp_mean > 0.0001:
                print(f'{self.feature_names[i]} ttest: {samp_ttest:.4f}')
                print(f'{self.feature_names[i]} true mean [margin of error]: {true_mean:.2f} [{samp_error[0]:.2f} - {samp_error[1]:.2f}]')
               
        self.background = background
        self.bkgoutcome = bkgoutcome
        self.bkgindices = bkgindices
        
        return background
    
    
    def _get_shapvalues(self, dataset):
        '''Calculate SHAP values of the XGBoost model for a given dataset.
        
        Parameters
        ----------
        dataset : str
            Dataset SHAP values to calculate.
        
        Returns
        -------
        Fitter
            Fitter with SHAP values calculated.
        '''
            
        import shap
        from shap.maskers import Independent
        
        if not hasattr(self, 'background'):
            background = self._set_background()
        
        else:
            background = self.background
        
        # Create masker for the background data. Otherwise SHAP limit background size
        # to 100 samples.
        mask = Independent(background, max_samples=background.shape[0])

        # Get features to make predictions
        features = self._getX('xg', dataset)
        feature_names = self.feature_names

        # Make probability predictions
        yprob = self.xg.predict_proba(features)[:, 1]

        # Create explainer
        explainer = shap.TreeExplainer(self.xg, data=mask, model_output='probability')

        # Explain predictions
        shap_values = explainer.shap_values(features)

        # Compute Marginal prediction
        marginal_pred = np.abs(shap_values.sum(1) + explainer.expected_value - yprob).max()
        if np.abs(marginal_pred) > 0.0001: print('Marginal prediction:', marginal_pred)

        # Get expected (base) value
        expected_value = explainer.expected_value

        explainer.values = shap_values
        explainer.data = features
        explainer.feature_names = feature_names
        
        if dataset == 'train':
            self.strain = explainer
            
        elif dataset == 'test':
            self.stest = explainer
            
        elif dataset == 'val':
            self.sval = explainer
            
        return self
    

    def rename(self, name):
        self.name = name
        return self
    
    
    def save(self, name=None):
        
        if name is None:
            savepath = f'saves/fitters/{self.name}-fitter.sav'
            
        else:
            savepath = f'saves/fitters/{name}-fitter.sav'
            
        pickle.dump(self, open(savepath, 'wb'))
        
        return self
    
    
    def get_features(self, *args, **kwargs):
        
        return get_features(self.fmatrix, *args, **kwargs)

        
    def split(self, test_size=None, n=None):
        '''Divide records into training and test datasets.
        
        Parameters
        ----------
        test_size : float, default None
            Proportion of the records to buil the test dataset. If None, it use the 20%
            of the records to build the test dataset. If test_size == 0, a mock test
            dataset with zero observations will be made. If test_size == 1, a mock 
            training dataset will be made.
        n : int, default None
            Substract random subsample from the training dataset.
            
        Return
        ------
        Fitter
            Fitter with divided datasets.
        '''
        
        import sklearn.model_selection
        
        records = self.records.copy()
        
        # Divide into features and outputs
        xseries = records.drop('death_hosp', axis=1)
        yseries = records['death_hosp']

        # Split into training and test sets
        if test_size is None:
            xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
                xseries.loc[xseries.index.str.startswith('02-')], 
                yseries.loc[yseries.index.str.startswith('02-')], 
                test_size=0.20, 
                random_state=0,
                stratify=yseries.loc[yseries.index.str.startswith('02-')]
            )
            self.xval = xseries.loc[xseries.index.str.startswith('01-')]
            self.yval = yseries.loc[yseries.index.str.startswith('01-')]
            
        elif test_size != 0 and test_size != 1:
            xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
                xseries, 
                yseries, 
                test_size=test_size, 
                random_state=0,
                stratify=yseries
            )
        elif test_size == 0:
            xtrain, ytrain = xseries, yseries
            xtest, ytest = pd.DataFrame(columns=xseries.columns), pd.Series(name=yseries.name)
            
        elif test_size == 1:
            xtrain, ytrain = pd.DataFrame(columns=xseries.columns), pd.Series(name=yseries.name)
            xtest, ytest = xseries, yseries
            

        # Substract subsample of training set
        if n:
            xtrain = xtrain.sample(n=n, random_state=0)
            ytrain = ytrain.sample(n=n, random_state=0)

        self.xseries = xseries
        self.yseries = yseries
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest

        return self
    
    
    def preprocess(self, restrict=None):
        '''Fit kernels for missing data imputation, standardization, and feature
        selection.
        
        Parameters
        ----------
        restrict : list, default None
            List of feature to restrict the model fit. If None, all features in the
            dataset will be used.
        
        Return
        ------
        Fitter
            Fitter with preprocesing kernel.
        '''
        
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.pipeline import Pipeline
        
        if not hasattr(self, 'xtrain') or not hasattr(self, 'ytrain'):
            self.split()
            
        # Imputation
        self.impute_kernel = Imputator()
        self.impute_kernel.fit(self.xtrain, self.ytrain)
        
        # Impute raw data
        self.itrain = self.impute_kernel.transform(self.xtrain)
        self.itest = self.impute_kernel.transform(self.xtest)
        
        if hasattr(self, 'xval'):
            self.ival = self.impute_kernel.transform(self.xval)
        
        # Quantile Transformation and feature selection
        # transform are the variables to be transformed and used in the models.
        # passthrough are the variables that are not transform, but used in the models.
        # variables that are not in transform or passthrough are dropped.
        transform = tuple(self.get_features('enabledToFit', 'numeric', restrict=restrict))
        passthrough = tuple(self.get_features('enabledToFit', '!numeric', restrict=restrict))
        
        transformer = QuantileTransformer(output_distribution='normal', random_state=0)
        self.quantile_kernel = ColumnTransformer(
            [
                ('quantileTransform', transformer, transform),
                ('passthrough', 'passthrough', passthrough),
            ],
            remainder='drop'
        )
        self.quantile_kernel.fit(self.xtrain, self.ytrain)
        
        # Feature selector
        # Same variables from the lsat step are used here.
        # This is a feature selector that filter the variables to be used in the model
        self.selector_kernel = ColumnTransformer(
            [
                ('passthrough', 'passthrough', transform + passthrough),
            ],
            remainder='drop'
        )
        
        self.selector_kernel.fit(self.xtrain, self.ytrain)
        self.feature_names = self.selector_kernel.get_feature_names()
        
        print(f'Constraining model fitting to {len(self.feature_names)} features.')
        
        # This attribute wraps the kernels to be used with the syntax of a sklearn pipeline
        self.pipeline = Pipeline([
            ('impute', self.impute_kernel),
            ('quantile', self.quantile_kernel),
            ('selector', self.selector_kernel)
        ])
            
        pathcv = f'saves/pipelines/{self.name}-pipeline.sav'
        pickle.dump(self.pipeline, open(pathcv, 'wb'))  
        
        return self
    
    
    def fitmodel(self, estimator=None, kfold=10):
        '''Fit models for a given estimator.
        
        Parameters
        ----------
        estimator : str or list, default None
            Estimator, or list of estimators to fit. If None, all available estimator
            will be fitted.
        kfold : int, default 10
            K-Fold used for crossvalidation.
        
        Return
        ------
        Fitter
            Fitter with fitted models.
        '''
        
        if not hasattr(self, 'pipeline'):
            self.preprocess()
            
        # Set classifier and cross-validation parameters
        if estimator is None:
            for name in self.available_models.keys():
                self.fitmodel(name, kfold)
                
            return self
        
        elif isinstance(estimator, list):
            for name in estimator:
                self.fitmodel(name, kfold)
                
            return self
                
        elif estimator == 'xg':
            from xgboost import XGBClassifier
            classifier = XGBClassifier()
            parameters = {
                'objective': ['binary:logistic'],
                'tree_method': ['approx'],
                'max_depth': [4],
                'max_delta_step': [1],
                'use_label_encoder': [False],
                'eval_metric': ['aucpr'],
                'random_state': [0],
                'verbosity': [0],
                'early_stopping_rounds': [10],
                'learning_rate': [0.05, 0.1],
                'gamma': [0.01, 0.05],
                'min_child_weight': [30, 40],
                'verbose': [0],
                'seed': [0]
            }

        elif estimator == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier()
            parameters = {
                'n_estimators': [100, 150, 200],
                'max_depth': [5, 6],
                'max_features': ['sqrt'],
                'class_weight': ['balanced'],
                'random_state': [0],
                'verbose': [0],
            }
            
        elif estimator == 'lr':
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression()
            parameters = {
                'penalty': ['l2'],
                'C': [0.5, 0.7, 1.0],
                'class_weight': ['balanced'],
                'solver': ['sag'],
                'max_iter': [5000],
                'random_state': [0],
                'verbose': [0],
            }
            
        elif estimator == 'nb':
            from sklearn.naive_bayes import GaussianNB
            classifier = GaussianNB()
            parameters = {}
            
        elif estimator == 'svm':
            # This model behave pretty similar to Linear Reg because we need 
            # to use a linear kernel to compensate for the dataset size.
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            classifier = CalibratedClassifierCV(LinearSVC())
            parameters = {
                'base_estimator__penalty': ['l2'],
                'base_estimator__C': [0.5, 0.7, 1.0],
                'base_estimator__class_weight': ['balanced'],
                'base_estimator__max_iter': [5000],
                'base_estimator__random_state': [0],
                'base_estimator__verbose': [0],
            }
            
        else:
            raise ValueError('Invalid cname. Check Fitter.available_models.')

            
        # Set variable method
        X = self._getX(estimator)
            
        # Just Fit!
        print(f'Fitting {self.available_models[estimator]}.')
        classifierCV = self._crossvalidator(classifier, parameters, kfold)
        classifierCV.fit(X, self.ytrain)
        
        best_estimator = classifierCV.best_estimator_
        best_estimator.feature_names = self.feature_names
        
        # Save model
        setattr(self, estimator + 'cv', classifierCV)
        setattr(self, estimator, best_estimator)
        
        pathcv = f'saves/models/crossvalidation/{self.name}-{estimator}-cv.sav'
        pathmd = f'saves/models/{self.name}-{estimator}.sav'
        
        pickle.dump(classifierCV, open(pathcv, 'wb'))        
        pickle.dump(best_estimator, open(pathmd, 'wb'))

        return self


    def get_feature_importance(self, estimator, n_iter=5, kfold=10):
        '''Compute feature importances. It use gain importance for XGBoost, coefficient 
        of the features for Logistic regression, and permutation importance for Random
        Forest and Naive Bayes.
        
        Parameters
        ----------
        estimator : str
            Estimator to calculate feature importances.
        n_iter : int, default 5
            Number of random shuffle iterations of the permutation importance.
            Decrease to improve speed, increase to get more precise estimates.
        kfold : int, default 10
            K-Fold used for crossvalidation of the permutation importance.
            
        Return
        ------
        Pandas.Series
            Feature importances of the selected estimator.
        '''
        from eli5.sklearn import PermutationImportance
        from sklearn.model_selection import StratifiedKFold
        
        if not hasattr(self, estimator): return False

        classifier = getattr(self, estimator)
        X = self._getX(estimator, 'train')
        y = self.ytrain

        if estimator in ['rf', 'nb']:
            perm = PermutationImportance(
                classifier,
                scoring='average_precision',
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0),
                random_state=0
            ).fit(X, y)
            classifier.perm_ = perm.feature_importances_
            feature_importances = perm.feature_importances_
            
        elif estimator == 'lr':
            feature_importances = classifier.coef_[0]

        else:
            feature_importances = classifier.feature_importances_

        return pd.Series(feature_importances, index=self.feature_names, name=self.available_estimators[estimator])

    
    def predict(self, estimator=None, dataset='val'):
        '''Make predictions of a model on a given dataset.
        
        Parameters
        ----------
        estimator : str or list, default None
            Estimator, or list of estimators to make predictions. If None, predictions for 
            all available models will be made.
        dataset : str, default 'val'
            Dataset to predict.
            
        Return
        ------
        Fitter
        '''
        
        if dataset == 'train' and not hasattr(self, 'ptrain'):
            self.ptrain = pd.DataFrame(index=self.xtrain.index)

        elif dataset == 'test' and not hasattr(self, 'ptest'):
            self.ptest = pd.DataFrame(index=self.xtest.index)
            
        elif dataset == 'val' and not hasattr(self, 'pval'):
            self.pval = pd.DataFrame(index=self.xval.index)

        if dataset == 'train':
            predictions = self.ptrain.copy()

        elif dataset == 'test':
            predictions = self.ptest.copy()
            
        elif dataset == 'val':
            predictions = self.pval.copy()

        else:
            raise ValueError('Choose between dataset \'train\', \'test\' or \'val\'.')
            

        if estimator is None:
            for estimator in self.available_estimators.keys():
                self.predict(estimator, dataset)
                
            return self
        
        elif isinstance(estimator, list):
            for name in estimator:
                self.predict(name, dataset)
                
            return self
        
        elif estimator in self.available_scores.keys():
            full_name = self.available_scores[estimator]
            X = self._getX(estimator, dataset)
            predictions[full_name] = predict_ccsm(estimator, X)
            
        elif estimator in self.available_models.keys():
            if not hasattr(self, estimator): 
                self.fitmodel(estimator)
            
            full_name = self.available_models[estimator]
            X = self._getX(estimator, dataset)
            predictions[full_name] = getattr(self, estimator).predict_proba(X)[:,1]
                        
        
        if dataset == 'train':
            self.ptrain = predictions
            
        elif dataset == 'test':
            self.ptest = predictions

        else:
            self.pval = predictions
            
        return self
    
    
    def ap_prc(self, estimator=None, dataset='val'):
        '''Calculate average precision (AP) of the precision-recall curve.
        
        Parameters
        ----------
        estimator : str or list, default None
            Estimator, or list of estimators to compute AP. If None, it will be made for 
            all available models.
        dataset : str, default 'val'
            Dataset to compute AP.
        
        Return
        ------
        dict
            AP of each estimator on the given dataset.
        '''
        
        if dataset == 'train' and not hasattr(self, 'ptrain'):
            self.predict(estimator, dataset)
            
        elif dataset == 'test' and not hasattr(self, 'ptest'):
            self.predict(estimator, dataset)
            
        elif dataset == 'val' and not hasattr(self, 'pval'):
            self.predict(estimator, dataset)
            
        
        if dataset == 'train':
            ytrue = self.ytrain.copy()
            predictions = self.ptrain.copy()
            
        elif dataset == 'test':
            ytrue = self.ytest.copy()
            predictions = self.ptest.copy()
            
        elif dataset == 'val':
            ytrue = self.yval.copy()
            predictions = self.pval.copy()
            
        else:
            raise ValueError('Choose between dataset \'train\', \'test\' or \'val\'.')
        
        # Drop rows with missing outputs or predictions
        predictions_dropna = predictions.dropna()
        
        # Force estimator as a list of estimator names
        if estimator is None:
            plot_list = list(self.available_estimators.values())
            
        elif isinstance(estimator, str):
            plot_list = [self.available_estimators[estimator]]
            
        elif isinstance(estimator, list):
            plot_list = []
            
            for name in estimator:
                plot_list.append(self.available_estimators[name])
            
        else:
            raise ValueError('Estimator must be a str, list, or None.')
        
        
        # Drop rows with missing outputs or predictions
        predictions_dropna = predictions[plot_list].dropna()
        ytrue = ytrue.loc[predictions_dropna.index]
        
        results = dict()
        for i, estimator_name in enumerate(plot_list):

            # Calculate progress.. It can take a while
            if estimator_name not in predictions_dropna.columns: continue
            progress = round(100 * i / len(plot_list), 2)
            
            ypred = predictions_dropna[estimator_name].values
            
            aps = sklearn.metrics.average_precision_score(ytrue, ypred)
            
            ci_lower, ci_upper = self._confidence_interval(ytrue, ypred, 'ap')
            
            results[estimator_name] = aps, ci_lower, ci_upper
        
        return results

    
    def shap_summary(self, dataset='test', limit=20, color_bar=False, refresh=False):
        import shap
        
        if dataset == 'train' and (not hasattr(self, 'strain') or refresh):
            self._get_shapvalues(dataset)
            
        elif dataset == 'test' and (not hasattr(self, 'stest') or refresh):
            self._get_shapvalues(dataset)
            
        elif dataset == 'val' and (not hasattr(self, 'sval') or refresh):
            self._get_shapvalues(dataset)
        
        
        if dataset == 'train':
            explainer = self.strain
            
        elif dataset == 'test':
            explainer = self.stest
            
        elif dataset == 'val':
            explainer = self.sval
            
        else:
            raise ValueError('Choose between dataset \'train\', \'test\' or \'val\'.')
            
        fnames = get_abbr(explainer.feature_names)
        
        figsize = tuple(plt.gcf().get_size_inches()) if len(plt.gcf().axes) > 0 else (9, 7)
        
        shap.summary_plot(
            explainer.values * 100, explainer.data, max_display=limit, feature_names=fnames, 
            cmap=plt.get_cmap(), plot_size=figsize, show=False, use_log_scale=False, color_bar=color_bar
        )

        fig = plt.gcf()
        ax = fig.axes[0]

        ax.set_xlabel('Impact on model output (%)')
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xlim(-40, 40)
        ax.tick_params(axis='y', which='major', pad=-20)
        ax.grid(True, axis='x')
        
        if color_bar: 
            fig.axes[1].set_ylabel('Predictor value', rotation=90)
            fig.axes[1].set_yticklabels(['Low', 'High'], rotation=90, va='center')
            ax.remove()
        fig.tight_layout()
        return fig


class Imputator(BaseEstimator, TransformerMixin):
        
        
    def __init__(self, fmatrix='base'):
        self.fmatrix = pd.read_csv(f'data/fmatrix-{fmatrix}.csv', index_col=0)
        self.imputable = list(self.get_features('imputable'))
    
    
    def get_features(self, *args, **kwargs):
        
        return get_features(self.fmatrix, *args, **kwargs)
    
    
    def fit(self, X, y=None):
        
        from miceforest import mean_match_default
        from miceforest import ImputationKernel
        
        self.impute_kernel = ImputationKernel(
            X,
            variable_schema=self.imputable,
            mean_match_scheme=mean_match_default.set_mean_match_candidates(5),
            datasets=1,
            train_nonmissing=True,
            random_state=0
        )
        self.impute_kernel.mice(3)
        
        return self
    
    
    def transform(self, X, y=None):
        # Manual imputation
        man_imputed = X.copy()
        man_imputed.loc[(man_imputed['FIO2 Miss'] == 1) & (man_imputed['Ventilation'] == 0), 'FIO2'] = 21
        
        # PMM imputation
        pmm_imputed = self.impute_kernel.impute_new_data(man_imputed, y).complete_data()
        
        return pmm_imputed
    
    
def get(dbname=None, ethnicity=None, los=30, miss_flag=True, name=None):
    '''Get data from a specific database and ethnicity.

    Parameters
    ----------
    dbname : str ('mimic', 'eicu'), default eicu
        Select database as source of information. 'eicu' selects data from eICU-CRD, and 'mimic'
        from MIMIC-III.
    ethnicity : str ('White', 'African American', 'Hispanic', 'Asian'), default None
        Select a specific ethnicity. If None, it will select all of them.
    los : int, default 30
        LOS survival threshold.
    miss_flag : bool, default True
        Add missing flags.
    name : str, default None
        Name of the resulting Fitter object.

    Returns
    -------
    Fitter
    '''
    import psycopg2 as pg
    
    # Construct query
    query = f'''select 
        -- Demographics
        idi, age as "Age", gender as "Gender", height as "Height", weight as "Weight", bmi as "BMI",

        -- Vital signs
        heartrate_avg as "Heart rate", resprate_avg as "Respiratory rate", temperature_avg as "Temperature", 
        sbp_avg as "SBP", dbp_avg as "DBP", map_avg as "MAP", cvp_avg as "CVP", spo2_avg as "SPO2", 
        fio2_avg as "FIO2", pao2_avg as "PAO2", paco2_avg as "PACO2", gcs_avg as "GCS",

        -- Lab tests
        bilirubin_avg as "Bilirubin", creatinine_avg as "Creatinine", hematocrit_avg as "Hematocrit",
        bun_avg as "BUN", platelets_avg as "Platelets", potassium_avg as "Potassium", sodium_avg as "Sodium", 
        chloride_avg as "Chloride", magnesium_avg as "Magnesium", wbc_avg as "WBC", 
        lymphocytes_avg as "Lymphocytes", ast_avg as "AST", albumin_avg as "Albumin", glucose_avg as "Glucose", 
        ptt_avg as "PTT", bnp_avg as "BNP", fibrinogen_avg as "Fibrinogen", hemoglobin_avg as "Hemoglobin", 
        lactate_avg as "Lactate", bicarbonate_avg as "Bicarbonate", ph_avg as "pH", 
        alp_avg as "Alkaline Phos.", urineoutput as "Urine output", 
        base_excess_avg as "Base excess", neutrophils_avg as "Neutrophils", alt_avg as "ALT",

        -- Comorbidities
        coalesce(sirs, 0) as "SIRS", 
        coalesce(cancer, 0) as "Cancer", 
        coalesce(lymphoma, 0) as "Lymphoma", 
        coalesce(rads, 0) as "Radiation", 
        coalesce(aids, 0) as "AIDS", 
        coalesce(hepatic_failure, 0) as "Hepatic failure", 
        coalesce(heart_failure, 0) as "Heart failure", 
        coalesce(respiratory_failure, 0) as "Respiratory failure", 
        coalesce(renal_failure, 0) as "Renal failure",

        -- Treatments
        coalesce(ventilation, 0) as "Ventilation", 
        coalesce(dopamine_avg, 0) as "Dopamine", 
        coalesce(epinephrine_avg, 0) as "Epinephrine", 
        coalesce(norepinephrine_avg, 0) as "Norepinephrine", 
        coalesce(dobutamine_avg, 0) as "Dobutamine", 

        -- Admission type and lenght of stay
        coalesce(surgery, 0) as "Admission for surgery", 
        coalesce(elective_surgery, 0) as "Adm. for elec. surgery", 
        case
            when los_preicu < 0 or los_preicu is null
                then 0
            else los_preicu
        end as "LOS preICU", 
        case
            when los_icu < {los} and death_hosp = 1
                then 1
            when los_icu < {los} and death_hosp = 0
                then 0
            else 0
        end as death_hosp
    from firstday
    where gender is not null and nft = 0'''

    if ethnicity is not None:
        query += f" and ethnicity = '{ethnicity}'"


    # Send query
    if dbname == 'eicu' or dbname is None:
        # Connect to database
        connection = pg.connect(user='postgres', password='postgres', host='localhost', dbname='eicu')

        # Set schema
        cursor = connection.cursor()
        cursor.execute('SET search_path TO aidxmods')
        connection.commit()
        cursor.close()

        eicu = pd.read_sql(query, connection).set_index('idi')

    if dbname == 'mimic' or dbname is None:
        # Connect to database
        connection = pg.connect(user='postgres', password='postgres', host='localhost', dbname='mimic')

        # Set schema
        cursor = connection.cursor()
        cursor.execute('SET search_path TO aidxmods')
        connection.commit()
        cursor.close()

        mimic = pd.read_sql(query, connection).set_index('idi')


    if dbname is None:
        fitter = Fitter(pd.concat([eicu, mimic]), name)

    elif dbname == 'eicu':
        fitter = Fitter(eicu, name)

    elif dbname == 'mimic':
        fitter = Fitter(mimic, name)

    else:
        fitter = Fitter(name=name)

    # Set flags of imputated values
    if miss_flag:

        # Reset flag list
        fitter.flags = set()

        # Select records and imputable columns
        records = fitter.records.copy()
        imputable = records[fitter.get_features('imputable')]

        for label in imputable.columns:

            # Address observations with missing values
            missing = imputable[label].isna()

            # Add missing flag for this feature
            flag_label = label + ' Miss'
            fitter.flags.add(flag_label)
            records[flag_label] = 0
            records.loc[missing, flag_label] = 1

        # Update intance records
        fitter.records = records

    return fitter


def load(name='default'):

    loadpath = f'saves/fitters/{name}-fitter.sav'

    return pickle.load(open(loadpath, 'rb'))


def get_fmatrix(fmatrix):
    '''Get a feature matrix from its name.'''
    
    return pd.read_csv(f'data/fmatrix-{fmatrix}.csv', index_col=0)


def get_features(fmatrix, *options, inverse=False, restrict=None):
    '''Get a list of features that satisties certain options.
    
    Parameters
    ----------
    fmatrix : str
        Feature matrix to use.
    *options : str
        Conditions to select features. For example "isNumeric" will select
        al features that are numeric. preppend an exclamation mark (such as 
        "!isNumeric") to select features that do not satisfies the condition.
        Multiple option arguments will yield in the intersection of the conditions.
    inverse : bool, default False
        Return the inverse selection of features.
    retrict : list, default None
        Restrict the reslting list of features into a subset of choice.
        
    Return
    ------
    List
        Selected features.
    '''

    if isinstance(fmatrix, str):
        selection = get_fmatrix(fmatrix).copy()
        
    else:
        selection = fmatrix.copy()

    for opt in options:
        if opt[:1] == "!":
            comparator = 0
            col = opt[1:]
        else:
            comparator = 1
            col = opt[0:]

        selection = selection[selection[col] == comparator]

    if inverse:
        selection = fmatrix.copy().drop(selection.index)

    if restrict:
        selection = selection.loc[selection.index.isin(restrict)]

    return selection.index


def rfe(fitter, n=25, step=0.15, n_iter=5, kfold=10):
    '''Select mot important features through RFE.
    
    Parameters
    ----------
    fitter : Fitter object
        Fitter to perform the RFE.
    n : int, default 25
        Features to select.
    step : int or float, default 015
        If int, the number of features to exclude on each selection step. In float, the
        proportion of features to exclude.
    n_iter : int, default 5
        Number of random shuffle iterations of the permutation importance.
        Decrease to improve speed, increase to get more precise estimates.
    kfold : int, default 10
        K-Fold used for crossvalidation of the permutation importance.
    
    Return
    ------
    Fitter
    '''
    from eli5.sklearn import PermutationImportance
    from sklearn import feature_selection
    from sklearn.model_selection import StratifiedKFold
    
    for key, name in fitter.available_models.items():
        
        print(name + ' RFE')
        estimator = getattr(fitter, key)
        X = fitter._getX(key, 'train')
        y = fitter.ytrain
        
        if key in ['rf', 'nb']:
            clf = PermutationImportance(
                estimator,
                scoring='average_precision',
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=kfold, shuffle=True, random_state=0),
                random_state=0
            )
            
        else:
            clf = estimator
        
        rfe = feature_selection.RFE(
            clf,
            n_features_to_select=n,
            step=step
        )
        rfe.fit(X, y)
        setattr(fitter, 'rfe' + key, rfe)

    return fitter


def most_important_features(fitter, limit=20, summary=False):
    '''Get the most important features from a model, grouping them into their
    corresponding parameters.
    
    Parameters
    ----------
    fitter : Fitter object
        Fitter to select the most important features for each model fitted.
    limit : int, default 20
        Number of distinct parameters admited.
    summary: bool, default False
        Return a summary with the feature selection process on each step: original list
        of features, primary selection through RFE, parameters selected from RFE, and
        features selected from RFE. Usefull only for debugging.
        
    Return
    ------
    dict
        Features selected from each model. If summary, return a summary with the feature 
        selection process on each step.
    '''
    
    selection = dict()
    summary_dict = dict()
    
    for key, name in fitter.available_models.items():
        rfe = getattr(fitter, 'rfe' + key)
        estimator = rfe.estimator_
        support = rfe.get_support()
        features = pd.Series(fitter.feature_names)[support]
        fnames = list(features.values)
        
        if "LogisticRegression" in str(estimator):
            feature_importances = estimator.coef_[0]

        else:
            feature_importances = estimator.feature_importances_
            
        sort_fscores = sorted(list(zip(fnames, feature_importances)), key=lambda x: x[1], reverse=True)
        
        imputable = fitter.get_features('imputable')
        missingFlag = fitter.get_features('missingFlag')
        select_features, select_parameters = list(), list()
        count = 0

        for fname, fscore in sort_fscores:

            if count == limit:
                break

            elif fname in select_features:
                continue

            elif fname in imputable:
                select_features.append(fname)
                select_features.append(fname + ' Miss')
                select_parameters.append(fname)

            elif fname in missingFlag:
                select_features.append(fname)
                select_features.append(fname.replace(' Miss', ''))
                select_parameters.append(fname.replace(' Miss', ''))

            elif fname in ['Adm. for elec. surgery', 'Admission for surgery']:
                select_features.append('Adm. for elec. surgery')
                select_features.append('Admission for surgery')
                select_parameters.append('Admission type')

            else:
                select_features.append(fname)
                select_parameters.append(fname)

            count += 1
            
        summary_dict[key] = {
            'original': fitter.feature_names, 
            'rfe_selection': fnames, 
            'parameters': select_parameters, 
            'features': select_features
        }

        selection[key] = select_features

    return summary_dict if summary else selection


def predict_ccsm(estimator, features):
    '''Predict classical clinical scores.'''
    
    if estimator == 'sofa':
        predictions = _estimator_sofa(features)

    elif estimator == 'oasis':
        predictions = _estimator_oasis(features)

    elif estimator == 'mods':
        predictions = _estimator_mods(features)

    elif estimator == 'apache':
        predictions = _estimator_apache(features)

    elif estimator == 'saps':
        predictions = _estimator_saps(features)
        
    return predictions / 100

    
def _estimator_sofa(features, only_preds=True):
    
    scores = pd.DataFrame(index=features.index)
    
    pao2 = features["PAO2"] # mmHg
    fio2 = features["FIO2"] / 100 # % to fraction
    vent = features["Ventilation"] # 0 - 1
    plt = features["Platelets"] # 10^3/uL
    gcs = features["GCS"] # 3 - 15
    br = features["Bilirubin"] # mg/dL
    map_ = features["MAP"] # mmHg
    dop = features["Dopamine"] # ug/kg/min
    epi = features["Epinephrine"] # ug/kg/min
    norepi = features["Norepinephrine"] # ug/kg/min
    cr = features["Creatinine"] # mg/dL
    uo = features["Urine output"] # mL/day
    dba = features["Dobutamine"] # ug/kg/min


    # Respiratory
    check = (pao2 / fio2 < 100) & (vent > 0)
    scores.loc[check, 'SOFA Score'] = 4
    check = ~(pao2 / fio2 < 100) & (pao2 / fio2 < 200) & (vent > 0)
    scores.loc[check, 'SOFA Score'] = 3
    check = ~(pao2 / fio2 < 200) & ~(vent > 0)
    scores.loc[check, 'SOFA Score'] = 2
    check = ~(pao2 / fio2 < 200) & (pao2 / fio2 < 300)
    scores.loc[check, 'SOFA Score'] = 2
    check = ~(pao2 / fio2 < 300) & (pao2 / fio2 < 400)
    scores.loc[check, 'SOFA Score'] = 1
    check = ~(pao2 / fio2 < 400)
    scores.loc[check, 'SOFA Score'] = 0

    # Coagulation
    check = (plt < 20)
    scores.loc[check, 'SOFA Score'] += 4
    check = (plt < 50) & ~(plt < 20)
    scores.loc[check, 'SOFA Score'] += 3
    check = (plt < 100) & ~(plt < 50)
    scores.loc[check, 'SOFA Score'] += 2
    check = (plt < 150) & ~(plt < 100)
    scores.loc[check, 'SOFA Score'] += 1
    check = ~(plt < 150)
    scores.loc[check, 'SOFA Score'] += 0

    # Neurologic
    check = (gcs < 6)
    scores.loc[check, 'SOFA Score'] += 4
    check = (gcs < 10) & ~(gcs < 6)
    scores.loc[check, 'SOFA Score'] += 3
    check = (gcs < 13) & ~(gcs < 10)
    scores.loc[check, 'SOFA Score'] += 2
    check = (gcs < 15) & ~(gcs < 13)
    scores.loc[check, 'SOFA Score'] += 1
    check = ~(gcs < 15)
    scores.loc[check, 'SOFA Score'] += 0

    # Hepatic
    check = (br >= 12.0)
    scores.loc[check, 'SOFA Score'] += 4
    check = (br >= 6.0) & ~(br >= 12.0)
    scores.loc[check, 'SOFA Score'] += 3
    check = (br >= 2.0) & ~(br >= 6.0)
    scores.loc[check, 'SOFA Score'] += 2
    check = (br >= 1.2) & ~(br >= 2.0)
    scores.loc[check, 'SOFA Score'] += 1
    check = ~(br >= 1.2)
    scores.loc[check, 'SOFA Score'] += 0



    # Cardiovascular
    check = (dop > 15) | (epi > 0.1) | (norepi > 0.1)
    scores.loc[check, 'SOFA Score'] += 4
    check = (
        ((dop > 5) & ~(dop > 15)) | 
        ((epi > 0) & ~(epi > 0.1)) | 
        ((norepi > 0) & ~(norepi > 0.1))
    )
    scores.loc[check, 'SOFA Score'] += 3
    check = (((dop > 0) & ~(dop > 5)) | (dba > 0))
    scores.loc[check, 'SOFA Score'] += 2
    check = (map_ < 70)
    scores.loc[check, 'SOFA Score'] += 1
    check = ~(map_ < 70)
    scores.loc[check, 'SOFA Score'] += 0

    # Renal
    check = (cr >= 5.0) | (uo < 200)
    scores.loc[check, 'SOFA Score'] += 4
    check = ((cr >= 3.5) & ~(cr >= 5.0)) | ((uo < 500) & ~(uo < 200))
    scores.loc[check, 'SOFA Score'] += 3
    check = (cr >= 2.0) & ~(cr >= 3.5)
    scores.loc[check, 'SOFA Score'] += 2
    check = (cr >= 1.2) & ~(cr >= 2.0)
    scores.loc[check, 'SOFA Score'] += 1
    check = ~(cr >= 1.2)
    scores.loc[check, 'SOFA Score'] += 0

    # Calculate mortality
    check = scores['SOFA Score'] < 2
    scores.loc[check, 'SOFA Pred'] = 0
    check = (scores['SOFA Score'] < 4) & ~(scores['SOFA Score'] < 2)
    scores.loc[check, 'SOFA Pred'] = 6.4
    check = (scores['SOFA Score'] < 6) & ~(scores['SOFA Score'] < 4)
    scores.loc[check, 'SOFA Pred'] = 20.2
    check = (scores['SOFA Score'] < 8) & ~(scores['SOFA Score'] < 6)
    scores.loc[check, 'SOFA Pred'] = 21.5
    check = (scores['SOFA Score'] < 10) & ~(scores['SOFA Score'] < 8)
    scores.loc[check, 'SOFA Pred'] = 33.3
    check = (scores['SOFA Score'] < 12) & ~(scores['SOFA Score'] < 10)
    scores.loc[check, 'SOFA Pred'] = 50.0
    check = (scores['SOFA Score'] < 15) & ~(scores['SOFA Score'] < 12)
    scores.loc[check, 'SOFA Pred'] = 95.2
    check = ~(scores['SOFA Score'] < 15)
    scores.loc[check, 'SOFA Pred'] = 95.2
    
    if only_preds:
        return scores['SOFA Pred']
    
    else:
        return scores


def _estimator_oasis(features, only_preds=True):
    
    scores = pd.DataFrame(index=features.index)

    los_preuci = features["LOS preICU"] * 24.0 # Days to Hours
    age = features["Age"] # Years
    gcs = features["GCS"] # 3 - 15
    hr = features["Heart rate"] # bpm
    map_ = features["MAP"] # mmHg
    rr = features["Respiratory rate"] # rpm
    temp = features["Temperature"] # °C
    uo = features["Urine output"] # mL/day
    vent = features["Ventilation"] # 0 - 1
    surgery  = features["Admission for surgery"] # 0 - 1
    elective_surgery = features["Adm. for elec. surgery"] # 0 - 1

    # Lenght of stay Pre-ICU
    check = (los_preuci < 0.17)
    scores.loc[check, 'OASIS Score'] = 5
    check = (los_preuci < 4.95) & ~(los_preuci < 0.17)
    scores.loc[check, 'OASIS Score'] = 3
    check = (los_preuci <= 24) & ~(los_preuci < 4.95)
    scores.loc[check, 'OASIS Score'] = 0
    check = (los_preuci <= 311.8) & ~(los_preuci <= 24)
    scores.loc[check, 'OASIS Score'] = 2
    check = ~(los_preuci <= 311.8)
    scores.loc[check, 'OASIS Score'] = 1

    # Age
    check = (age < 24)
    scores.loc[check, 'OASIS Score'] += 0
    check = (age <= 53) & ~(age < 24)
    scores.loc[check, 'OASIS Score'] += 3
    check = (age <= 77) & ~(age <= 53)
    scores.loc[check, 'OASIS Score'] += 6
    check = (age <= 89) & ~(age <= 77)
    scores.loc[check, 'OASIS Score'] += 9
    check = ~(age <= 89)
    scores.loc[check, 'OASIS Score'] += 1

    # Glasgow Coma Score
    check = (gcs <= 7)
    scores.loc[check, 'OASIS Score'] += 10
    check = (gcs <= 13) & ~(gcs <= 7)
    scores.loc[check, 'OASIS Score'] += 4
    check = (gcs <= 14) & ~(gcs <= 13)
    scores.loc[check, 'OASIS Score'] += 3
    check = ~(gcs <= 14)
    scores.loc[check, 'OASIS Score'] += 0

    # Hearth rate
    check = (hr < 33)
    scores.loc[check, 'OASIS Score'] += 4
    check = (hr <= 88) & ~(hr < 33)
    scores.loc[check, 'OASIS Score'] += 0
    check = (hr <= 106) & ~(hr <= 88)
    scores.loc[check, 'OASIS Score'] += 1
    check = (hr <= 125) & ~(hr <= 106)
    scores.loc[check, 'OASIS Score'] += 3
    check = ~(hr <= 125)
    scores.loc[check, 'OASIS Score'] += 6

    # Mean Arterial Pressure
    check = (map_ < 20.65)
    scores.loc[check, 'OASIS Score'] += 4
    check = (map_ <= 51) & ~(map_ < 20.65)
    scores.loc[check, 'OASIS Score'] += 3
    check = (map_ <= 61.33) & ~(map_ <= 51)
    scores.loc[check, 'OASIS Score'] += 2
    check = (map_ <= 143.44) & ~(map_ <= 61.33)
    scores.loc[check, 'OASIS Score'] += 0
    check = ~(map_ <= 143.44)
    scores.loc[check, 'OASIS Score'] += 3

    # Respiratory rate
    check = (rr < 6)
    scores.loc[check, 'OASIS Score'] += 10
    check = (rr < 13) & ~(rr < 6)
    scores.loc[check, 'OASIS Score'] += 1
    check = (rr < 23) & ~(rr < 13)
    scores.loc[check, 'OASIS Score'] += 0
    check = (rr < 31) & ~(rr < 23)
    scores.loc[check, 'OASIS Score'] += 1
    check = (rr < 44) & ~(rr < 31)
    scores.loc[check, 'OASIS Score'] += 6
    check = ~(rr < 44)
    scores.loc[check, 'OASIS Score'] += 9

    # Temperature
    check = (temp < 33.22)
    scores.loc[check, 'OASIS Score'] += 3
    check = (temp < 35.94) & ~(temp < 33.22)
    scores.loc[check, 'OASIS Score'] += 4
    check = (temp < 36.40) & ~(temp < 35.94)
    scores.loc[check, 'OASIS Score'] += 2
    check = (temp < 36.86) & ~(temp < 36.40)
    scores.loc[check, 'OASIS Score'] += 0
    check = (temp < 39.88) & ~(temp < 36.86)
    scores.loc[check, 'OASIS Score'] += 2
    check = ~(temp < 39.88)
    scores.loc[check, 'OASIS Score'] += 6

    # Urinary output
    check = (uo < 671)
    scores.loc[check, 'OASIS Score'] += 10
    check = (uo < 1427) & ~(uo < 671)
    scores.loc[check, 'OASIS Score'] += 5
    check = (uo < 2544) & ~(uo < 1427)
    scores.loc[check, 'OASIS Score'] += 1
    check = (uo < 6896) & ~(uo < 2544)
    scores.loc[check, 'OASIS Score'] += 0
    check = ~(uo < 6896)
    scores.loc[check, 'OASIS Score'] += 8

    # Ventilation
    check = (vent > 0)
    scores.loc[check, 'OASIS Score'] += 9
    check = ~(vent > 0)
    scores.loc[check, 'OASIS Score'] += 0

    # Elective Surgery
    check = (elective_surgery > 0)
    scores.loc[check, 'OASIS Score'] += 0
    check = ~(elective_surgery > 0)
    scores.loc[check, 'OASIS Score'] += 9

    # Calculate mortality
    scores["OASIS Pred"] = round(scores["OASIS Score"] * 1.07, 1)

    if only_preds:
        return scores['OASIS Pred']
    
    else:
        return scores

def _estimator_mods(features, only_preds=True):
    
    scores = pd.DataFrame(index=features.index)

    pao2 = features["PAO2"] # mmHg
    fio2 = features["FIO2"] / 100 # % to fraction
    plt = features["Platelets"] # 10^3/uL
    br = features["Bilirubin"]  # mg/dL
    hr = features["Heart rate"] # bpm
    cvp = features["CVP"] # mmHg
    map_ = features["MAP"] # mmHg
    gcs = features["GCS"]
    cr = features["Creatinine"] # mg/dL

    # Respiratory
    check = (pao2 / fio2 <= 75)
    scores.loc[check, 'MODS Score'] = 4
    check = (pao2 / fio2 <= 150) & ~(pao2 / fio2 <= 75)
    scores.loc[check, 'MODS Score'] = 3
    check = (pao2 / fio2 <= 225) & ~(pao2 / fio2 <= 150)
    scores.loc[check, 'MODS Score'] = 2
    check = (pao2 / fio2 <= 300) & ~(pao2 / fio2 <= 225)
    scores.loc[check, 'MODS Score'] = 1
    check = ~(pao2 / fio2 <= 300)
    scores.loc[check, 'MODS Score'] = 0

    # Coagulation
    check = (plt <= 20)
    scores.loc[check, 'MODS Score'] += 4
    check = (plt <= 50) & ~(plt <= 20)
    scores.loc[check, 'MODS Score'] += 3
    check = (plt <= 80) & ~(plt <= 50)
    scores.loc[check, 'MODS Score'] += 2
    check = (plt <= 120) & ~(plt <= 80)
    scores.loc[check, 'MODS Score'] += 1
    check = ~(plt <= 120)
    scores.loc[check, 'MODS Score'] += 0

    # Hepatic
    check = (br > 14.0)
    scores.loc[check, 'MODS Score'] += 4
    check = (br > 7.0) & ~(br > 14.0)
    scores.loc[check, 'MODS Score'] += 3
    check = (br > 3.5) & ~(br > 7.0)
    scores.loc[check, 'MODS Score'] += 2
    check = (br > 1.2) & ~(br > 3.5)
    scores.loc[check, 'MODS Score'] += 1
    check = ~(br > 1.2)
    scores.loc[check, 'MODS Score'] += 0

    # Neurologic
    check = (gcs <= 6)
    scores.loc[check, 'MODS Score'] += 4
    check = (gcs <= 9) & ~(gcs <= 6)
    scores.loc[check, 'MODS Score'] += 3
    check = (gcs <= 12) & ~(gcs <= 9)
    scores.loc[check, 'MODS Score'] += 2
    check = (gcs <= 14) & ~(gcs <= 12)
    scores.loc[check, 'MODS Score'] += 1
    check = ~(gcs <= 14)
    scores.loc[check, 'MODS Score'] += 0

    # Cardiovascular
    check = (hr * cvp / map_ > 30)
    scores.loc[check, 'MODS Score'] += 4
    check = (hr * cvp / map_ > 20) & ~(hr * cvp / map_ > 30)
    scores.loc[check, 'MODS Score'] += 3
    check = (hr * cvp / map_ > 15) & ~(hr * cvp / map_ > 20)
    scores.loc[check, 'MODS Score'] += 2
    check = (hr * cvp / map_ > 10) & ~(hr * cvp / map_ > 15)
    scores.loc[check, 'MODS Score'] += 1
    check = ~(hr * cvp / map_ > 10)
    scores.loc[check, 'MODS Score'] += 0

    # Renal (umol/L)
    check = (cr > 5.65)
    scores.loc[check, 'MODS Score'] += 4
    check = (cr > 3.95) & ~(cr > 5.65)
    scores.loc[check, 'MODS Score'] += 3
    check = (cr > 2.26) & ~(cr > 3.95)
    scores.loc[check, 'MODS Score'] += 2
    check = (cr > 1.13) & ~(cr > 2.26)
    scores.loc[check, 'MODS Score'] += 1
    check = ~(cr > 1.13)
    scores.loc[check, 'MODS Score'] += 0

    # Calculate mortality
    check = scores['MODS Score'] <= 0
    scores.loc[check, 'MODS Pred'] = 0
    check = (scores['MODS Score'] <= 4) & ~(scores['MODS Score'] <= 0)
    scores.loc[check, 'MODS Pred'] = 2.0
    check = (scores['MODS Score'] <= 8) & ~(scores['MODS Score'] <= 4)
    scores.loc[check, 'MODS Pred'] = 5.0
    check = (scores['MODS Score'] <= 12) & ~(scores['MODS Score'] <= 8)
    scores.loc[check, 'MODS Pred'] = 25.0
    check = (scores['MODS Score'] <= 16) & ~(scores['MODS Score'] <= 12)
    scores.loc[check, 'MODS Pred'] = 50.0
    check = (scores['MODS Score'] <= 20) & ~(scores['MODS Score'] <= 16)
    scores.loc[check, 'MODS Pred'] = 75.0
    check = ~(scores['MODS Score'] <= 20)
    scores.loc[check, 'MODS Pred'] = 100.0

    if only_preds:
        return scores['MODS Pred']
    
    else:
        return scores


def _estimator_apache(features, only_preds=True):
    
    scores = pd.DataFrame(index=features.index)

    temp = features["Temperature"] # °C
    map_ = features["MAP"] # mmHg
    hr = features["Heart rate"] # bpm
    rr = features["Respiratory rate"] # rpm
    pao2 = features["PAO2"] # mmHg
    fio2 = features["FIO2"] # (%)
    paco2 = features["PACO2"] # mmHg
    ph = features["pH"]
    sna = features["Sodium"] # mmol/L
    sk = features["Potassium"] # mmol/L
    cr = features["Creatinine"] # mg/dL
    ht = features["Hematocrit"] # (%)
    wbc = features["WBC"] # 10^3/uL
    gcs = features["GCS"] # 3 - 15
    shco3 = features["Bicarbonate"] # mmol/L
    age = features["Age"] # Years
    hepatic_failure = features["Hepatic failure"] # 0 - 1
    heart_failure = features["Heart failure"] # 0 - 1
    respiratory_failure = features["Respiratory failure"] # 0 - 1
    renal_failure = features["Renal failure"] # 0 - 1
    cancer = features["Cancer"] # 0 - 1
    lymphoma = features["Lymphoma"] # 0 -1 
    aids = features["AIDS"] # 0 - 1
    rads = features["Radiation"] # 0 - 1
    surgery  = features["Admission for surgery"] # 0 - 1
    elective_surgery = features["Adm. for elec. surgery"] # 0 - 1
    no_abg_available = (fio2.isna()) | (pao2.isna()) | (paco2.isna())
    AaDO2 = fio2 * 710.0 - paco2 * 1.25 - pao2
    multiplier = renal_failure + 1
    chronic = (
        (hepatic_failure == 1) | (heart_failure == 1) | (respiratory_failure == 1)
        | (renal_failure == 1) | (cancer == 1) | (lymphoma == 1) | (aids == 1) | (rads == 1)
    )

    # Temperature
    check = (temp < 30) | (temp >= 41)
    scores.loc[check, 'APACHE II Score'] = 4
    check = ((temp < 32) & ~(temp < 30)) | ((temp >= 39) & ~(temp >= 41))
    scores.loc[check, 'APACHE II Score'] = 3
    check = (temp < 34) & ~(temp < 32)
    scores.loc[check, 'APACHE II Score'] = 2
    check = ((temp < 36) & ~(temp < 34)) | ((temp >= 38.5) & ~(temp >= 39))
    scores.loc[check, 'APACHE II Score'] = 1
    check = (temp >= 36) & (temp < 38.5)
    scores.loc[check, 'APACHE II Score'] = 0

    # Mean Arterial Pressure
    check = (map_ < 50) | (map_ >= 160)
    scores.loc[check, 'APACHE II Score'] += 4
    check = (map_ >= 130) & ~(map_ >= 160)
    scores.loc[check, 'APACHE II Score'] += 3
    check = ((map_ < 70) & ~(map_ < 50)) | ((map_ >= 110) & ~(map_ >= 130))
    scores.loc[check, 'APACHE II Score'] += 2
    check = (map_ > 70) & (map_ < 110)
    scores.loc[check, 'APACHE II Score'] += 0

    # Hearth rate
    check = (hr < 40) | (hr >= 180)
    scores.loc[check, 'APACHE II Score'] += 4
    check = ((hr < 55) & ~(hr < 40)) | ((hr >= 140) & ~(hr >= 180))
    scores.loc[check, 'APACHE II Score'] += 3
    check = ((hr < 70) & ~(hr < 55)) | ((hr >= 110) & ~(hr >= 140))
    scores.loc[check, 'APACHE II Score'] += 2
    check = (hr > 70) & (hr < 110)
    scores.loc[check, 'APACHE II Score'] += 0

    # Respiratory rate
    check = (rr < 6) | (rr >= 50)
    scores.loc[check, 'APACHE II Score'] += 4
    check = (rr >= 35) & ~(rr >= 50)
    scores.loc[check, 'APACHE II Score'] += 3
    check = (rr < 10) & ~(rr < 6)
    scores.loc[check, 'APACHE II Score'] += 2
    check = ((rr < 12) & ~(rr < 10)) | ((rr >= 25) & ~(rr >= 35))
    scores.loc[check, 'APACHE II Score'] += 1
    check = (rr >= 12) & (rr < 25)
    scores.loc[check, 'APACHE II Score'] += 0

    # Oxygenation
    check = ((fio2 >= 0.5) & (AaDO2 >= 500)) | ((fio2 < 0.5) & (pao2 < 55))
    scores.loc[check, 'APACHE II Score'] += 4
    check = ((fio2 >= 0.5) & (AaDO2 >= 350) & ~(AaDO2 >= 500)) | ((fio2 < 0.5) & (pao2 < 61) & ~(pao2 < 55))
    scores.loc[check, 'APACHE II Score'] += 3
    check = ((fio2 >= 0.5) & (AaDO2 >= 200) & ~(AaDO2 >= 350))
    scores.loc[check, 'APACHE II Score'] += 2
    check = ((fio2 < 0.5) & (pao2 <= 70) & ~(pao2 < 61))
    scores.loc[check, 'APACHE II Score'] += 1
    check = ((fio2 >= 0.5) & ~(AaDO2 >= 200)) | ((fio2 < 0.5) & ~(pao2 <= 70))
    scores.loc[check, 'APACHE II Score'] += 0

    # Arterial pH
    check = (ph < 7.15) | (ph >= 7.7)
    scores.loc[check, 'APACHE II Score'] += 4
    check = ((ph < 7.25) & ~(ph < 7.15)) | ((ph >= 7.6) & ~(ph >= 7.7))
    scores.loc[check, 'APACHE II Score'] += 3
    check = (ph < 7.33) & ~(ph < 7.25)
    scores.loc[check, 'APACHE II Score'] += 2
    check = (ph >= 7.5) & ~(ph >= 7.6)
    scores.loc[check, 'APACHE II Score'] += 1
    check = (ph >= 7.33) & (ph < 7.5)
    scores.loc[check, 'APACHE II Score'] += 0

    # Serum Na
    check = (sna <= 110) | (sna >= 180)
    scores.loc[check, 'APACHE II Score'] += 4
    check = ((sna < 120) & ~(sna <= 110)) | ((sna >= 160) & ~(sna >= 180))
    scores.loc[check, 'APACHE II Score'] += 3
    check = ((sna < 130) & ~(sna < 120)) | ((sna >= 155) & ~(sna >= 160))
    scores.loc[check, 'APACHE II Score'] += 2
    check = (sna >= 150) & ~(sna >= 155)
    scores.loc[check, 'APACHE II Score'] += 1
    check = (sna >= 130) & (sna < 150)
    scores.loc[check, 'APACHE II Score'] += 0

    # Serum K
    check = (sk < 2.5) | (sk >= 7)
    scores.loc[check, 'APACHE II Score'] += 4
    check = (sk >= 6) & ~(sk >= 7)
    scores.loc[check, 'APACHE II Score'] += 3
    check = (sk < 3) & ~(sk < 2.5)
    scores.loc[check, 'APACHE II Score'] += 2
    check = (sk < 3.5) & ~(sk < 3)
    scores.loc[check, 'APACHE II Score'] += 1
    check = (sk < 6) & (sk >= 3.5)
    scores.loc[check, 'APACHE II Score'] += 0

    # Cretinine
    check = (cr > 3.45)
    scores.loc[check, 'APACHE II Score'] += 4 * multiplier
    check = (cr >= 1.92) & ~(cr > 3.45)
    scores.loc[check, 'APACHE II Score'] += 3 * multiplier
    check = (cr < 0.60) | ((cr >= 1.47) & ~(cr >= 1.92))
    scores.loc[check, 'APACHE II Score'] += 2 * multiplier
    check = (cr >= 0.60) & (cr < 1.47)
    scores.loc[check, 'APACHE II Score'] += 0

    # Hematocrit
    check = (ht < 20) | (ht >= 60)
    scores.loc[check, 'APACHE II Score'] += 4
    check = (ht < 30) & ~(ht < 20)
    scores.loc[check, 'APACHE II Score'] += 2
    check = (ht >= 50) & ~(ht >= 60)
    scores.loc[check, 'APACHE II Score'] += 2
    check = (ht >= 46) & ~(ht >= 50)
    scores.loc[check, 'APACHE II Score'] += 1
    check = (ht < 46) & (ht >= 30)
    scores.loc[check, 'APACHE II Score'] += 0

    # WBC
    check = (wbc < 1) | (wbc >= 40)
    scores.loc[check, 'APACHE II Score'] += 4
    check = (wbc < 3) & ~(wbc < 1)
    scores.loc[check, 'APACHE II Score'] += 2
    check = (wbc >= 20) & ~(wbc >= 40)
    scores.loc[check, 'APACHE II Score'] += 2
    check = (wbc >= 15) & ~(wbc >= 20)
    scores.loc[check, 'APACHE II Score'] += 1
    check = (ht < 15) & (ht >= 3)
    scores.loc[check, 'APACHE II Score'] += 0

    # Glasgow Coma Score
    scores.loc[check, 'APACHE II Score'] += 15 - gcs

    # Serum HCO3
    check = no_abg_available & (shco3 < 15)
    scores.loc[check, 'APACHE II Score'] += 4
    check = no_abg_available & (shco3 >= 52)
    scores.loc[check, 'APACHE II Score'] += 4
    check = no_abg_available & ((shco3 < 18) & ~(shco3 < 15))
    scores.loc[check, 'APACHE II Score'] += 3
    check = no_abg_available & ((shco3 >= 41) & ~(shco3 >= 52))
    scores.loc[check, 'APACHE II Score'] += 3
    check = no_abg_available & ((shco3 < 22) & ~(shco3 < 18))
    scores.loc[check, 'APACHE II Score'] += 2
    check = no_abg_available & ((shco3 >= 32) & ~(shco3 >= 41))
    scores.loc[check, 'APACHE II Score'] += 1
    check = no_abg_available & (shco3 < 32) & (shco3 >= 22)
    scores.loc[check, 'APACHE II Score'] += 0

    # Age
    check = (age >= 75)
    scores.loc[check, 'APACHE II Score'] += 6
    check = (age >= 65) & ~(age >= 75)
    scores.loc[check, 'APACHE II Score'] += 5
    check = (age >= 55) & ~(age >= 65)
    scores.loc[check, 'APACHE II Score'] += 3
    check = (age > 44) & ~(age >= 55)
    scores.loc[check, 'APACHE II Score'] += 2
    check = ~(age > 44)
    scores.loc[check, 'APACHE II Score'] += 0

    # Chronic diseases and admission
    check = chronic & (elective_surgery == 1)
    scores.loc[check, 'APACHE II Score'] += 2
    check = chronic & ~(elective_surgery == 1)
    scores.loc[check, 'APACHE II Score'] += 5
    check = ~chronic & ~(elective_surgery == 1)
    scores.loc[check, 'APACHE II Score'] += 0

    # Calcular nivel de riesgo
    logit = -3.517 + scores["APACHE II Score"] * 0.146
    scores["APACHE II Pred"] = round(100 * np.exp(logit) / (1 + np.exp(logit)), 1)

    if only_preds:
        return scores['APACHE II Pred']
    
    else:
        return scores


def _estimator_saps(features, only_preds=True):
    
    scores = pd.DataFrame(index=features.index)

    age = features["Age"] # Years
    hr = features["Heart rate"] # bpm
    sbp = features["SBP"] # mmHg
    temp = features["Temperature"] # °C
    gcs = features["GCS"] # 3 - 15
    pao2 = features["PAO2"] # mmHg
    fio2 = features["FIO2"] # (%)
    vent = features["Ventilation"] # 0 - 2
    bun = features["BUN"] # mg/dL
    uo = features["Urine output"] # mL/day
    sna = features["Sodium"] # mmol/L
    sk = features["Potassium"] # mmol/L
    shco3 = features["Bicarbonate"] # mmol/L
    br = features["Bilirubin"] # mg/dL
    wbc = features["WBC"] # 10^3/uL
    cancer = features["Cancer"] # 0 - 1
    lymphoma = features["Lymphoma"] # 0 - 1
    aids = features["AIDS"] # 0 - 1
    surgery  = features["Admission for surgery"] # 0 - 1
    elective_surgery  = features["Adm. for elec. surgery"] # 0 - 1


    # Age
    check = (age >= 80)
    scores.loc[check, 'SAPS II Score'] = 18
    check = (age >= 75) & ~(age >= 80)
    scores.loc[check, 'SAPS II Score'] = 16
    check = (age >= 70) & ~(age >= 75)
    scores.loc[check, 'SAPS II Score'] = 15
    check = (age >= 60) & ~(age >= 70)
    scores.loc[check, 'SAPS II Score'] = 12
    check = (age >= 40) & ~(age >= 60)
    scores.loc[check, 'SAPS II Score'] = 7
    check = ~(age >= 40)
    scores.loc[check, 'SAPS II Score'] = 0

    # HR
    check = (hr < 40)
    scores.loc[check, 'SAPS II Score'] += 11
    check = (hr < 70) & ~(hr < 40)
    scores.loc[check, 'SAPS II Score'] += 2
    check = (hr < 120) & ~(hr < 70)
    scores.loc[check, 'SAPS II Score'] += 0
    check = (hr < 160) & ~(hr < 120)
    scores.loc[check, 'SAPS II Score'] += 4
    check = ~(hr < 160)
    scores.loc[check, 'SAPS II Score'] += 7

    # SBP
    check = (sbp < 70)
    scores.loc[check, 'SAPS II Score'] += 13
    check = (sbp < 100) & ~(sbp < 70)
    scores.loc[check, 'SAPS II Score'] += 5
    check = (sbp < 200) & ~(sbp < 100)
    scores.loc[check, 'SAPS II Score'] += 0
    check = ~(sbp < 200)
    scores.loc[check, 'SAPS II Score'] += 2

    # Temperature
    check = (temp > 39)
    scores.loc[check, 'SAPS II Score'] += 3
    check = ~(temp > 39)
    scores.loc[check, 'SAPS II Score'] += 0

    # Glasgow Coma Score
    check = (gcs < 6)
    scores.loc[check, 'SAPS II Score'] += 26
    check = (gcs < 9) & ~(gcs < 6)
    scores.loc[check, 'SAPS II Score'] += 13
    check = (gcs < 11) & ~(gcs < 9)
    scores.loc[check, 'SAPS II Score'] += 7
    check = (gcs < 14) & ~(gcs < 11)
    scores.loc[check, 'SAPS II Score'] += 5
    check = ~(gcs < 14)
    scores.loc[check, 'SAPS II Score'] += 0

    # Oxygenation
    check = (pao2 / fio2 < 100) & (vent > 0)
    scores.loc[check, 'SAPS II Score'] += 11
    check = (pao2 / fio2 < 200) & ~(pao2 / fio2 < 100) & (vent > 0)
    scores.loc[check, 'SAPS II Score'] += 9
    check = (pao2 / fio2 >= 200) & (vent > 0)
    scores.loc[check, 'SAPS II Score'] += 6
    check = ~(vent > 0)
    scores.loc[check, 'SAPS II Score'] += 0

    # BUN
    check = (bun >= 84)
    scores.loc[check, 'SAPS II Score'] += 10
    check = (bun >= 28) & ~(bun >= 84)
    scores.loc[check, 'SAPS II Score'] += 6
    check = ~(bun >= 28)
    scores.loc[check, 'SAPS II Score'] += 0

    # Urine output
    check = (uo < 500)
    scores.loc[check, 'SAPS II Score'] += 11
    check = (uo < 1000) & ~(uo < 500)
    scores.loc[check, 'SAPS II Score'] += 4
    check = ~(uo < 1000)
    scores.loc[check, 'SAPS II Score'] += 0

    # Sodio
    check = (sna < 125)
    scores.loc[check, 'SAPS II Score'] += 5
    check = (sna < 145) & ~(sna < 125)
    scores.loc[check, 'SAPS II Score'] += 0
    check = ~(sna < 145)
    scores.loc[check, 'SAPS II Score'] += 1

    # Potasio
    check = (sk < 3)
    scores.loc[check, 'SAPS II Score'] += 3
    check = (sk < 5) & ~(sk < 3)
    scores.loc[check, 'SAPS II Score'] += 0
    check = ~(sk < 5)
    scores.loc[check, 'SAPS II Score'] += 3

    # HCO3
    check = (shco3 < 15)
    scores.loc[check, 'SAPS II Score'] += 6
    check = (shco3 < 20) & ~(shco3 < 15)
    scores.loc[check, 'SAPS II Score'] += 3
    check = ~(shco3 < 20)
    scores.loc[check, 'SAPS II Score'] += 0

    # Bilirubin
    check = (br < 4)
    scores.loc[check, 'SAPS II Score'] += 0
    check = (br < 6) & ~(br < 4)
    scores.loc[check, 'SAPS II Score'] += 4
    check = ~(br < 6)
    scores.loc[check, 'SAPS II Score'] += 9

    # WBC
    check = (wbc < 1)
    scores.loc[check, 'SAPS II Score'] += 12
    check = (wbc < 20) & ~(wbc < 1)
    scores.loc[check, 'SAPS II Score'] += 0
    check = ~(wbc < 20)
    scores.loc[check, 'SAPS II Score'] += 3

    # Chronic disease
    check = (aids == 1)
    scores.loc[check, 'SAPS II Score'] += 17
    check = (lymphoma == 1) & ~(aids == 1)
    scores.loc[check, 'SAPS II Score'] += 10
    check = (cancer == 1) & ~(lymphoma == 1) & ~(aids == 1)
    scores.loc[check, 'SAPS II Score'] += 9

    # Admission type
    check = (elective_surgery == 1)
    scores.loc[check, 'SAPS II Score'] += 0
    check = (surgery == 1) & ~(elective_surgery == 1)
    scores.loc[check, 'SAPS II Score'] += 8
    check = ~(surgery == 1) & ~(elective_surgery == 1)
    scores.loc[check, 'SAPS II Score'] += 6

    # Set mortality rate
    logit = -7.7631 + 0.0737 * scores["SAPS II Score"] + 0.9971 * np.log(scores["SAPS II Score"] + 1)
    scores["SAPS II Pred"] = round(100 * np.exp(logit) / (1 + np.exp(logit)), 1)

    if only_preds:
        return scores['SAPS II Pred']
    
    else:
        return scores   


def cluster_matrix(array, inplace=False):
    '''Rearranges the correlation matrix, array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    '''
    import pandas as pd
    import numpy as np
    import scipy
    import scipy.stats as stats
    import scipy.cluster.hierarchy as sch

    pairwise_distances = sch.distance.pdist(array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        array = array.copy()
    
    if isinstance(array, pd.DataFrame):
        return array.iloc[idx, :].T.iloc[idx, :]
    
    return array[idx, :][:, idx]

def plot_matrix(matrix, title, save=None, show=True):
    '''Plot correlation or covariance matrix.'''
    import matplotlib.pyplot as plt
    import matplotlib.cm
    import matplotlib.colors
    import numpy as np
    
    names = matrix.columns
    cmap = matplotlib.cm.get_cmap('bwr', 6)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    
    fig, axs = plt.subplots(figsize=(20, 20))
    
    im = axs.imshow(matrix, cmap=cmap, norm=norm)
    axs.set_title(title)
    axs.set_xticks(np.arange(len(names)))
    axs.set_xticklabels(names, rotation=90)
    axs.set_yticks(np.arange(len(names)))
    axs.set_yticklabels(names)
    axs.tick_params(axis='both', bottom=False, labelbottom=False, top=True, labeltop=True)
    cbar = axs.figure.colorbar(im, ax=axs)
    cbar.ax.set_ylabel('Color map', rotation=-90, va="bottom")
    
    handlefig(save, show)


def get_abbr(labels, fmatrix='base'):
    abbr = get_fmatrix(fmatrix)['abbr']

    if isinstance(labels, str): return abbr[labels] if labels in abbr.keys() else ''

    new_names = list()
    
    for label in labels:
                
        if label in abbr.keys():
            new_names.append(abbr[label])
        else:
            new_names.append(label)
            
    return new_names


def save_pickle(stuff, path):
    import pickle
    pickle.dump(stuff, open(path, 'wb'))
    
    
def load_pickle(path):
    import pickle
    return pickle.load(open(path, 'rb'))


def get_barplot_offset(i, N):
    return (-1 / 2) + (2 *  i + 1) / (2 * N)

def feature_attr(features, attr):
    
    dictionary = {
        'vent': ['Ventilation', 'Y/N'],
        'lactate': ['Lactate', 'mg/dL'],
        'elective_surgery': ['Adm. for elect. surgery', 'Y/N'],
        'bun': ['BUN', 'mg/mL'],
        'gcs': ['GCS', ''],
        'fio2': ['FiO2', '%'],
        'age': ['Age', 'years'],
        'uo': ['Urine output', 'mL/24h'],
        'rr': ['Respiratory rate', 'RPM'],
        'hr': ['Heart rate', 'BPM'],
        'temp': ['Temperature', '°C'],
        'sirs': ['SIRS', 'Y/N'],
        'surgery': ['Admission for Surgery', 'Y/N'],
        'spo2': ['SpO2', '%'],
        'ast': ['AST', 'U/L'],
        'wbc': ['WBC', '10^3/mm^2'],
        'los_preuci': ['LOS pre-ICU admission', 'hours'],
        'cancer': ['Cancer', 'Y/N'],
        'sbp': ['SBP', 'mmHg'],
        'cvp': ['CVP', 'mmHg'],
        'paco2': ['PaCO2', 'mmHg'],
        'ptt': ['PTT', 's'],
        'plt': ['Platelets', '10^3/mm^2'],
        'bnp': ['BNP', 'pg/mL'],
        'ph': ['pH', ''],
        'lymphoma': ['Lymphoma', 'Y/N'],
        'alp': ['ALP', 'U/L'],
        'be': ['Base Excess', 'mEq/L'],
        'norepi': ['Norepinephrine', 'mcg/kg/min'],
        'cr': ['Creatinine', 'mg/mL'],
        'dbp': ['DBP', 'mmHg'],
        'bmi': ['BMI', 'kg/m^2'],
        'weight': ['Weight', 'kg'],
        'hemoglobin': ['Hemoglobin', 'g/dL'],
        'br': ['Bilirrubin', 'mg/mL'],
        'alb': ['Albumin', 'g/dL'],
        'map': ['MAP', 'mmHg'],
        'ht': ['Hematocrit', '%'],
        'neutrophils': ['Neutrophils', '%'],
        'epi': ['Epinephrine', 'mcg/kg/min'],
        'pao2': ['PaO2', 'mmHg'],
        'glc': ['Glucose', 'mg/mL'],
        'chloride': ['Chloride', 'mEq/L'],
        'sna': ['Sodium', 'mEq/L'],
        'heart_failure': ['Heart Failure', 'Y/N'],
        'lymphocytes': ['Lymphocytes', '%'],
        'hepatic_failure': ['Hepatic Failure', 'Y/N'],
        'magnesium': ['Magnesium', 'mEq/L'],
        'dop': ['Dopamine', 'mcg/kg/min'],
        'sk': ['Potassium', 'mEq/L'],
        'alt': ['ALT', 'U/L'],
        'shco3': ['Blood Bicarbonate', 'mEq/L'],
        'fg': ['Fibrinogen', 'mg/dL'],
        'height': ['Height', 'm'],
        'sex': ['Gender', 'F/M'],
        'aids': ['AIDS', 'Y/N'],
        'respiratory_failure': ['Respiratory Failure', 'Y/N'],
        'rads': ['Immunosupression bc. radiation', 'Y/N'],
        'renal_failure': ['Renal Failure', 'Y/N'],
        'dba': ['Dobutamine', 'mcg/kg/min'],
    }
    
    labels = list()
    
    for feature in features:
        if attr == 'name':
            labels.append(dictionary[feature][0])
        elif attr == 'uom':
            labels.append(dictionary[feature][1])
    
    return labels
    
def norm_shap_value(value, values):
    pivot = - values[values < 0].min() if value < 0 else values[values > 0].max()
    return 0.5 * (value / pivot + 1)

    
def set_prop_cycle(theme, series):
    
    from cycler import cycler
    
    if theme == 'gs':
        color_cycle = GS_color_cycle
        hatch_cycle = GS_hatch_cycle
        linestyle_cycle = GS_linestyle_cycle
        linewidth_cycle = GS_linewidth_cycle
        cmap = GS_cmap
        
    else:
        color_cycle = CB_color_cycle
        hatch_cycle = CB_hatch_cycle
        linestyle_cycle = CB_linestyle_cycle
        linewidth_cycle = CB_linewidth_cycle
        cmap = CB_cmap
    
    if series < 1 or series > 10:
        raise ValueError('series must be between 1 and 10')
        
    elif series <= 5:
        color = cycler(color=color_cycle[:series])
        hatch = cycler(hatch=hatch_cycle[:1] * series)
        linestyle = cycler(linestyle=linestyle_cycle[:1] * series)
        linewidth = cycler(linewidth=linewidth_cycle[:1] * series)
        
    elif series % 2 == 0 or series % 5 == 0:
        partial_series = series // 2
        color = cycler(color=color_cycle[:partial_series] * 2)
        hatch = cycler(hatch=hatch_cycle[:1] * partial_series + hatch_cycle[1:2] * partial_series)
        linestyle = cycler(linestyle=linestyle_cycle[:1] * partial_series + linestyle_cycle[1:2] * partial_series)
        linewidth = cycler(linewidth=linewidth_cycle[:1] * series)
        
    elif series % 3 == 0:
        partial_series = series // 3
        color = cycler(color=color_cycle[:partial_series] * 2)
        hatch = cycler(hatch=hatch_cycle[:1] * partial_series + hatch_cycle[1:2] * partial_series)
        linestyle = cycler(linestyle=linestyle_cycle[:1] * partial_series + linestyle_cycle[1:2] * partial_series)
        linewidth = cycler(linewidth=linewidth_cycle[:1] * series)
        
    elif series % 7 == 0:
        color = cycler(color=color_cycle[:5] + color_cycle[:2])
        hatch = cycler(hatch=hatch_cycle[:1] * 5 + hatch_cycle[1:2] * 2)
        linestyle = cycler(linestyle=linestyle_cycle[:1] * 5 + linestyle_cycle[1:2] * 2)
        linewidth = cycler(linewidth=linewidth_cycle[:1] * series)
        
    return color + linestyle + linewidth, cmap
    
        
def setfig(nrows=1, ncols=1, theme='gs', nseries=5, **kwargs):
    
    fig, axs = plt.subplots(nrows, ncols, **kwargs)
    
    cycler, cmap = set_prop_cycle(theme, nseries)
    plt.set_cmap(cmap)
    
    if nrows > 1 and ncols > 1:
        for i in range(0, nrows):
            for j in range(0, ncols):
                axs[i, j].set_prop_cycle(cycler)
    
    elif nrows > 1 or ncols > 1:
        limit = nrows if nrows > 1 else ncols
        for i in range(0, limit):
                axs[i].set_prop_cycle(cycler)
    
    else:
        axs.set_prop_cycle(cycler)
    
    return fig, axs

        
def handlefig(save=False, show=True):
    
    plt.tight_layout()
    if save:
        plt.savefig(f'figures/{save}.png')
        plt.savefig(f'figures/svg/{save}.svg')
    if show:
        plt.show()
    plt.close()
    

# Grayscale plot theme
GS_color_cycle = [
    '#dddddd', 
    '#bbbbbb', 
    '#888888', 
    '#555555',
    '#000000'
]

GS_hatch_cycle = ['', '///', 'xxx']

GS_linestyle_cycle = ['-', '--', '-.']

GS_linewidth_cycle = [2]

GS_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('customgrayscale', ['#cccccc', '#000000'], 1000)
matplotlib.cm.register_cmap(name='customgrayscale', cmap=GS_cmap)

# Color blind friendly plot theme
# Modified from Paul Tol's Vibrant palette
CB_color_cycle = [
    '#000000', # BLACK
    '#0077BB', # BLUE
    '#EE7733', # ORANGE
    '#117733', # GREEN
    '#882255', # WINE
]

CB_hatch_cycle = ['', '///', 'xxx']

CB_linestyle_cycle = ['-', '--', '-.']

CB_linewidth_cycle = [2]

CB_cmap = matplotlib.cm.get_cmap('cool')
