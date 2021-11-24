from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

gs_clf = 0


# Use to search for proper model parameters
# Parameters:
# text_clf_svm: Pipeline for Model
# data_frame: Our dataframe with features
# training_label: Our training label list
def search_for_parameters(text_clf_svm, data_frame, training_label, parameters):
    global gs_clf
    print("Starting GridSearchCV...")
    start = datetime.now()
    gs_clf = GridSearchCV(text_clf_svm, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(data_frame, training_label)
    print("Best Score: " + str(gs_clf.best_score_))
    print("Best Params: \n")
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    end = datetime.now()
    duration = end - start
    print("GridSearchCV duration: " + str(duration))
    print(
        "The best parameters are %s with a score of %0.2f"
        % (gs_clf.best_params_, gs_clf.best_score_)
    )


def get_plot(gamma_range, C_range):
    scores = gs_clf.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot
    )
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title("Accuracy")
    plt.show()