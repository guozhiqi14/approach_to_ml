import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

"""
在贪心特征选择算法中，每次迭代都需要遍历所有特征，以便找到当前迭代中最能提高模型性能的特征。这是贪心算法的核心思想：在每一步选择当前最优的选项，而不考虑全局最优。

具体来说，以下是每次迭代遍历所有特征的原因和逻辑：

详细解释
寻找当前最优特征：

在每次迭代中，算法需要找到一个特征，该特征在加入到当前已选择的特征集合后，能够最大程度地提高模型的性能。
为了找到这个特征，必须评估每一个尚未被选择的特征，看看它们在当前上下文中的表现如何。
评估每个特征的贡献：

对于每个特征，算法将其加入到当前已选择的特征集合中，形成一个新的特征子集。
然后，使用这个新的特征子集训练模型，并评估模型的性能（例如，通过计算AUC）。
通过这种方式，算法可以确定每个特征在当前上下文中的贡献。
选择最佳特征：

在遍历所有特征并评估它们的贡献后，算法选择贡献最大的特征，将其加入到已选择的特征集合中。
这个过程重复进行，直到满足停止条件（例如，模型性能不再提高）。


在每次迭代中，for feature in range(num_features) 确实会遍历所有特征（0 到 3）。
但是，通过 if feature in good_features: continue 这行代码，我们会跳过已经被选择的特征。因此，虽然我们遍历了所有特征，但实际上只会评估那些尚未被选择的特征。


"""
class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection.
    You will need to modify it quite a bit to make it suitable for your dataset.
    """

    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns Area Under ROC Curve (AUC)
        NOTE: We fit the data and calculate AUC on same data. WE ARE OVERFITTING HERE.
        But this is also a way to achieve greedy selection. k-fold will take k times longer.
        If you want to implement it in really correct way, calculate OOF AUC and return mean AUC over k folds.
        This requires only a few lines of change and has been shown a few times in this book.
        :param X: training data
        :param y: targets
        :return: overfitted area under the roc curve
        """
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def _feature_selection(self, X, y):
        """
        This function does the actual greedy selection
        :param X: data, numpy array
        :param y: targets, numpy array
        :return: (best scores, best features)
        """
        good_features = []
        best_scores = []

        num_features = X.shape[1]

        while True:
            this_feature = None
            best_score = 0

            for feature in range(num_features):
                if feature in good_features:
                    continue

                selected_features = good_features + [feature]
                xtrain = X[:, selected_features]
                score = self.evaluate_score(xtrain, y)

                if score > best_score:
                    this_feature = feature
                    best_score = score

            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)

            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break

        # return best scores and good features
        # why do we remove the last data point?
        # This condition is checking if the last score in best_scores is less than the second last score. If it is,
        # then the while loop is broken. This is a stopping criterion for the greedy feature selection process.
        # The reason for this is because the greedy feature selection process is iterative and adds one feature at a time
        # to the model. It calculates the score (in this case, AUC) after each addition. If the addition of a new feature does not
        # improve the score (i.e., the latest score is less than the previous score), it indicates that the newly added feature is
        # not contributing positively to the model. Therefore, the process stops, effectively not including the last feature in the "good features" list.
        # This is a common approach in greedy algorithms to prevent overfitting and to ensure that only beneficial
        # features are included in the model.
        return best_scores[:-1], good_features[:-1]

    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments
        """
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores


if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=100)
    X_transformed, scores = GreedyFeatureSelection()(X, y)